"""yaml loader: merge layered yamls (via OmegaConf) + dataclass validation.

Order: dataclass defaults → case yaml → overlays (CLI order) → `--set`
dotlist → in-memory extra_yaml.

Note: `_base/*.yaml` base-layer selection and `${env:VAR}` env-var
interpolation are not currently implemented (YAML is loaded with
resolve=False).
"""

from __future__ import annotations

import dataclasses
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any, Sequence

_logger = logging.getLogger(__name__)

import yaml
from locomgcm.constants import ROTATION_PERIOD_EARTH
from locomgcm.external_paths import resolve_input_path, resolve_input_references

try:
    from omegaconf import OmegaConf
    _HAVE_OMEGACONF = True
except ImportError:
    _HAVE_OMEGACONF = False

from ._double_track import mirror_double_track
from ._schema import (
    AdjointConfig,
    AdvectionConfig,
    BoundaryConditionManifest,
    BoundaryConfig,
    CacheConfig,
    CaseConfig,
    CheckpointConfig,
    CompareConfig,
    DebugConfig,
    DynamicsEntry,
    FreeSurfaceConfig,
    GridConfig,
    ImplicitVerticalAdvectionConfig,
    InitialConditionConfig,
    MitPickupConfig,
    MitComparisonConfig,
    MitParallelConfig,
    ModelManifest,
    NonhydrostaticConfig,
    ObservationsConfig,
    OperatorSpec,
    OutputConfig,
    StationOutputConfig,
    PhysicsConfig,
    PipelineSpec,
    RunConfig,
    RunIdentity,
    SectionTransportUConfig,
    SourceSpec,
    StepperManifest,
    TimeConfig,
    TracerConfig,
)


def _read_yaml(path: Path) -> dict:
    if _HAVE_OMEGACONF:
        cfg = OmegaConf.load(path)
        return OmegaConf.to_container(cfg, resolve=False)  # resolve at merge time
    else:
        with open(path) as f:
            return yaml.safe_load(f) or {}


# A key of the form ``<list>[<name>]`` addresses the entry of a list of named
# mappings (``model.dynamics``) whose ``name`` equals ``<name>``.
_NAMED_ENTRY_KEY = re.compile(r"^(\w+)\[([^\]]+)\]$")


def _entry_names(entries: Any) -> list[Any]:
    if not isinstance(entries, list):
        return []
    return [e.get("name") for e in entries if isinstance(e, dict)]


def _merge_named_entry(entries: Any, list_key: str, selector: str, patch: Any) -> list:
    if not isinstance(entries, list):
        raise ValueError(
            f"'{list_key}[{selector}]' addresses an entry of '{list_key}', "
            "which is not a list in the merged configuration"
        )
    out = list(entries)
    for i, entry in enumerate(out):
        if isinstance(entry, dict) and entry.get("name") == selector:
            out[i] = _deep_merge(entry, patch)
            return out
    raise ValueError(
        f"'{list_key}' has no entry named {selector!r}; known names: {_entry_names(entries)}"
    )


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursive dict merge: overlay values win.

    Mappings merge key by key. A list is replaced as a whole, except that a key
    ``<list>[<name>]`` merges into the entry of that name. A mapping written
    over a list of named entries is an error, because it would silently replace
    the list instead of selecting a named entry.
    """
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return overlay
    out = dict(base)
    for k, v in overlay.items():
        named = _NAMED_ENTRY_KEY.match(str(k))
        if named is not None:
            list_key, selector = named.groups()
            out[list_key] = _merge_named_entry(out.get(list_key), list_key, selector, v)
        elif k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        elif k in out and isinstance(out[k], list) and isinstance(v, dict) and _entry_names(out[k]):
            raise ValueError(
                f"'{k}' is a list of named entries; address one as '{k}[<name>]' "
                f"(known names: {_entry_names(out[k])}) or supply the whole list"
            )
        else:
            out[k] = v
    return out


# Split a dotted key on dots outside square brackets, so that
# ``model.dynamics[horizontal_viscosity].kwargs.viscosity`` keeps its selector.
_DOT_OUTSIDE_BRACKETS = re.compile(r"\.(?![^\[]*\])")


def _parse_set_value(raw: str) -> Any:
    """Scalar typing for ``--set`` values: int, float, bool, null, then text.

    Bracketed or braced text is read as a YAML list or mapping.
    """
    text = raw.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if text[:1] in {"[", "{"}:
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError:
            return raw
    return raw


def _parse_set_dotlist(items: Sequence[str]) -> dict:
    """Parse ``--set key.sub.path=value`` items into a nested dict."""
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set item {item!r} is not of the form key.path=value")
        key, _, raw_val = item.partition("=")
        parts = _DOT_OUTSIDE_BRACKETS.split(key.strip())
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = _parse_set_value(raw_val)
    return out


def _warn_if_days_overridden_by_steps(
    sets: Sequence[str],
    time_cfg: "TimeConfig",
) -> None:
    """Emit a logger.warning when --set time.days= is silently swallowed.

    Operators familiar with older yamls (which used ``days`` as the primary field)
    sometimes write ``--set time.days=0.5`` to shorten a run, but if the
    yaml already contains ``time.steps: N`` the ``--set`` is a silent
    no-op — ``compute_n_steps`` short-circuits on ``steps_override`` and
    the supplied ``days`` value is never consulted.

    Detection: any item in ``sets`` that starts with ``time.days`` (after
    stripping leading ``time.`` path components) AND the resolved
    ``TimeConfig.steps`` is not None (or ``seconds`` is not None, meaning
    a higher-precision override is already active).

    No behaviour change: this is warn-only.
    """
    if not sets:
        return
    active_override = time_cfg.steps is not None or time_cfg.seconds is not None
    if not active_override:
        return
    # Match any dotlist item whose key is exactly "time.days" (the driver
    # passes it as "time.days=<value>" after converting SECS= → time.steps=
    # itself, so the only dangerous pattern is a literal `--set time.days=`).
    _TIME_DAYS_PAT = re.compile(r"^time\.days\s*=")
    for item in sets:
        if _TIME_DAYS_PAT.match(item.strip()):
            if time_cfg.steps is not None:
                winner = f"time.steps={time_cfg.steps}"
            else:
                winner = f"time.seconds={time_cfg.seconds}"
            _logger.warning(
                "run config: --set time.days= was supplied but is IGNORED "
                "because %s has higher precedence. "
                "Use STEPS=<int> or SECS=<float> env knobs (or --set %s) "
                "to override run length.",
                winner,
                "time.steps=<int>" if time_cfg.steps is not None else "time.seconds=<float>",
            )
            break  # one warning is enough


def load_run_config(
    config: Path,
    overlays: Sequence[Path] = (),
    sets: Sequence[str] = (),
    extra_yaml: dict | None = None,
) -> RunConfig:
    """Load + merge + validate.

    Args:
      config: case yaml (required)
      overlays: list of overlay yamls applied in order
      sets: list of `key.path=value` dotlist overrides
      extra_yaml: in-memory dict applied last (before env interpolation)

    Returns:
      Frozen RunConfig validated via __post_init__.
    """
    if config is not None and not Path(config).exists():
        raise FileNotFoundError(f"run config file not found: {config}")
    run_cfg = _dict_to_run_config(merge_yaml_layers(config, overlays, sets, extra_yaml))
    _warn_if_days_overridden_by_steps(sets, run_cfg.time)
    return run_cfg


def merge_yaml_layers(
    config: Path | str | None,
    overlays: Sequence[Path | str] = (),
    sets: Sequence[str] = (),
    extra_yaml: dict | None = None,
) -> dict:
    """Return merged YAML with external input references resolved.

    Layers after the case file are overrides: a parameter the case writes under
    both ``physics.*`` and ``model.*`` follows an override of either spelling
    (see ``_double_track``).
    """
    merged: dict = {}
    if config is not None and Path(config).exists():
        merged = _deep_merge(merged, _read_yaml(Path(config)))
    for o in overlays:
        merged = _apply_override(merged, _read_yaml(Path(o)))
    if sets:
        merged = _apply_override(merged, _parse_set_dotlist(sets))
    if extra_yaml is not None:
        merged = _apply_override(merged, extra_yaml)
    return resolve_input_references(merged)


def _apply_override(merged: dict, layer: dict) -> dict:
    return mirror_double_track(layer, _deep_merge(merged, layer))


def _reject_unknown_keys(
    section: str, mapping: dict, schema: type, loader_keys: tuple[str, ...] = ()
) -> None:
    """Refuse a yaml key that neither ``schema`` nor the loader reads.

    ``loader_keys`` are the spellings the loader reads besides the dataclass
    fields (aliases and shortcuts it desugars).  Any other key would be
    dropped without effect.
    """
    known = {field.name for field in dataclasses.fields(schema)} | set(loader_keys)
    unknown = sorted(set(mapping) - known)
    if unknown:
        raise ValueError(f"{section} has unknown key(s): {', '.join(unknown)}")


def _dict_to_run_config(d: dict) -> RunConfig:
    """Convert raw merged dict → typed RunConfig (frozen dataclass tree)."""
    _reject_unknown_keys("the run config", d, RunConfig, ("time_step",))
    # These inputs bypass SourceSpec. Prepared-run paths keep their own owner.
    for section, fields in (
        (d.get("grid", {}) or {}, ("cube_grid_dir", "cube_bathy_path")),
        ((d.get("case", {}) or {}).get("extras", {}) or {},
         ("cube_ic_npz", "cube_forcing_zarr")),
        (d.get("initial_condition", {}) or {}, ("woa23_dir",)),
        ((d.get("initial_condition", {}) or {}).get("mit_pickup", {}) or {}, ("meta", "data", "cd_data")),
    ):
        for field in fields:
            if section.get(field):
                section[field] = str(resolve_input_path(section[field]))
    schema_version = d.get("schema_version", "0.1.0")

    model_raw = d.get("model", {}) or {}
    stepper_kwargs_raw = dict(
        (model_raw.get("stepper", {}) or {}).get("kwargs", {}) or {}
    )
    implicit_vertical_advection_keys = (
        "temperature_implicit_vertical_advection",
        "salinity_implicit_vertical_advection",
        "momentum_implicit_vertical_advection",
    )
    duplicate_sources = [
        f"model.stepper.kwargs.{key}"
        for key in implicit_vertical_advection_keys
        if key in stepper_kwargs_raw
    ]
    for index, entry in enumerate(model_raw.get("dynamics", []) or []):
        if not isinstance(entry, dict):
            continue
        module_name = str(entry.get("name", ""))
        kwargs = dict(entry.get("kwargs", {}) or {})
        duplicate_sources.extend(
            f"model.dynamics[{index}:{module_name}].kwargs.{key}"
            for key in implicit_vertical_advection_keys
            if key in kwargs
        )
    if duplicate_sources:
        raise ValueError(
            "implicit vertical advection has duplicate derived source(s): canonical "
            "physics.implicit_vertical_advection and "
            + ", ".join(duplicate_sources)
        )

    # Case kind
    case_d = d.get("case", {}) or {}
    _reject_unknown_keys("case", case_d, CaseConfig)
    case = CaseConfig(
        kind=str(case_d.get("kind", "regional")),
        mit_verification_dir=case_d.get("mit_verification_dir"),
        extras=dict(case_d.get("extras", {}) or {}),
    )

    # Run identity
    run_d = d.get("run", {})
    _reject_unknown_keys("run", run_d, RunIdentity)
    start_date_raw = run_d.get("start_date", "2019-01-01")
    if isinstance(start_date_raw, str):
        start_date_val = date.fromisoformat(start_date_raw)
    else:
        start_date_val = start_date_raw
    run = RunIdentity(
        experiment_name=run_d.get("experiment_name", "default"),
        start_date=start_date_val,
        run_id=run_d.get("run_id"),
        device=run_d.get("device", "auto"),
        preprocess=run_d.get("preprocess", True),
        precision=str(run_d.get("precision", "float64")),
    )

    # Grid
    g = d.get("grid", {})
    _reject_unknown_keys("grid", g, GridConfig)
    vlt_raw = g.get("vertical_layer_thicknesses", [])
    grid = GridConfig(
        nx=int(g.get("nx", 1)),
        ny=int(g.get("ny", 1)),
        nz=int(g.get("nz", 1)),
        center_lat=float(g.get("center_lat", 0.0)),
        center_lon=float(g.get("center_lon", 0.0)),
        resolution_deg=float(g.get("resolution_deg", 1.0)),
        depth_m=float(g.get("depth_m", 100.0)),
        vertical_layer_thicknesses=tuple(float(v) for v in vlt_raw) if vlt_raw else (float(g.get("depth_m", 100.0)),),
        vertical_center_spacing=tuple(float(v) for v in g.get("vertical_center_spacing", ())),
        kind=str(g.get("kind", "latlon")),
        cube_face_n=(int(g["cube_face_n"]) if g.get("cube_face_n") is not None else None),
        cube_grid_dir=(str(g["cube_grid_dir"]) if g.get("cube_grid_dir") is not None else None),
        cube_grid_stem=(str(g["cube_grid_stem"]) if g.get("cube_grid_stem") is not None else None),
        cube_bathy_path=(str(g["cube_bathy_path"]) if g.get("cube_bathy_path") is not None else None),
        cube_advection_topology=(
            str(g["cube_advection_topology"])
            if g.get("cube_advection_topology") is not None
            else None
        ),
        cube_tile_nx=(int(g["cube_tile_nx"]) if g.get("cube_tile_nx") is not None else None),
        cube_tile_ny=(int(g["cube_tile_ny"]) if g.get("cube_tile_ny") is not None else None),
        cube_tile_halo=(
            int(g["cube_tile_halo"]) if g.get("cube_tile_halo") is not None else None
        ),
    )

    # Time
    t = _deep_merge(d.get("time", {}), d.get("time_step", {}) or {})
    _reject_unknown_keys("time (with time_step)", t, TimeConfig)
    time_cfg = TimeConfig(
        dt=float(t.get("dt", 60.0)),
        dt_mom=float(t.get("dt_mom", t.get("dt", 60.0))),
        dt_freesurf=float(t.get("dt_freesurf", t.get("dt", 60.0))),
        dt_tracer=float(t.get("dt_tracer", t.get("dt", 60.0))),
        dt_clock=float(t.get("dt_clock", t.get("dt", 60.0))),
        days=float(t.get("days", 1.0)),
        steps=(int(t["steps"]) if t.get("steps") is not None else None),
        # Preferred yaml horizon field. Accepts
        # int|float|str; strings are evaluated by TimeConfig.__post_init__
        # via the safe arithmetic evaluator (e.g. ``"5 * 86400"`` → 432000.0).
        seconds=t.get("seconds"),
    )

    # MitParallel
    mp = d.get("mit_parallel", {})
    _reject_unknown_keys("mit_parallel", mp, MitParallelConfig)
    mit_par = MitParallelConfig(
        nPx=int(mp.get("nPx", 1)),
        nPy=int(mp.get("nPy", 1)),
        sNx=mp.get("sNx"),
        sNy=mp.get("sNy"),
        OLx=int(mp.get("OLx", 4)),
        OLy=int(mp.get("OLy", 4)),
    )

    # Physics + nested
    p = d.get("physics", {})
    _reject_unknown_keys("physics", p, PhysicsConfig)
    nh_d = p.get("nonhydrostatic", {})
    _reject_unknown_keys("physics.nonhydrostatic", nh_d, NonhydrostaticConfig)
    fs_d = p.get("free_surface", {})
    _reject_unknown_keys("physics.free_surface", fs_d, FreeSurfaceConfig)
    tracer_d = p.get("tracer", {}) or {}
    _reject_unknown_keys("physics.tracer", tracer_d, TracerConfig)
    if "implicit_vertical_advection" not in p:
        implicit_vertical_advection = ImplicitVerticalAdvectionConfig()
    else:
        iva_d = p["implicit_vertical_advection"]
        if not isinstance(iva_d, dict):
            raise ValueError(
                "physics.implicit_vertical_advection must be a mapping with "
                "temperature, salinity, and momentum bool leaves"
            )
        _reject_unknown_keys(
            "physics.implicit_vertical_advection", iva_d, ImplicitVerticalAdvectionConfig
        )
        iva_values: dict[str, bool] = {}
        for name in ("temperature", "salinity", "momentum"):
            value = iva_d.get(name, False)
            if type(value) is not bool:
                raise ValueError(
                    "physics.implicit_vertical_advection."
                    f"{name} must be bool, got {type(value).__name__}"
                )
            iva_values[name] = value
        implicit_vertical_advection = ImplicitVerticalAdvectionConfig(**iva_values)
    physics = PhysicsConfig(
        reference_density=(
            float(p["reference_density"])
            if p.get("reference_density") is not None
            else None
        ),
        nonlin_free_surf=int(p.get("nonlin_free_surf", 0)),
        enable_vertical_shear=bool(p.get("enable_vertical_shear", False)),
        split_coriolis=bool(p.get("split_coriolis", False)),
        nonhydrostatic=NonhydrostaticConfig(
            enabled=bool(nh_d.get("enabled", False)),
            cg3d_max_iters=int(nh_d.get("cg3d_max_iters", 150)),
            cg3d_target_residual=float(nh_d.get("cg3d_target_residual", 1e-7)),
            cg3d_target_res_w_unit=float(nh_d.get("cg3d_target_res_w_unit", -1.0)),
            nh_am2=float(nh_d.get("nh_am2", 1.0)),
            select_3d_cori_scheme=int(nh_d.get("select_3d_cori_scheme", 1)),
            select_cori_map=int(nh_d.get("select_cori_map", -1)),
            f_prime=float(nh_d.get("f_prime", 0.0)),
        ),
        free_surface=FreeSurfaceConfig(
            select_r_star=int(fs_d.get("select_r_star", 0)),
            lin_fs_conserve_tr=bool(fs_d.get("lin_fs_conserve_tr", False)),
        ),
        viscosity=float(p.get("viscosity", 0.0)),
        viscosity_scheme=str(p.get("viscosity_scheme", "constant")),
        visc_c2_leith=float(p.get("visc_c2_leith", 0.0)),
        visc_c2_smag=float(p.get("visc_c2_smag", 0.0)),
        diffusivity=float(p.get("diffusivity", 0.0)),
        implicit_viscosity=(
            None
            if p.get("implicit_viscosity") is None
            else float(p["implicit_viscosity"])
        ),
        implicit_vertical_viscosity=(
            None
            if p.get("implicit_vertical_viscosity") is None
            else p["implicit_vertical_viscosity"]
        ),
        no_slip_sides=bool(p.get("no_slip_sides", False)),
        tracer=TracerConfig(
            implicit_vertical_diffusion=(
                None
                if tracer_d.get("implicit_vertical_diffusion") is None
                else bool(tracer_d["implicit_vertical_diffusion"])
            ),
            diffusivity_kr=(None if tracer_d.get("diffusivity_kr") is None
                            else float(tracer_d["diffusivity_kr"])),
        ),
        implicit_vertical_advection=implicit_vertical_advection,
        ke_scheme=int(p.get("ke_scheme", 0)),
        rotation_period=float(p.get("rotation_period", ROTATION_PERIOD_EARTH)),
        # 
        advective_form=bool(p.get("advective_form", False)),
        cfl_safety=float(p.get("cfl_safety", 0.5)),
        hfac_min_dz=(float(p["hfac_min_dz"]) if p.get("hfac_min_dz") is not None else None),
        fast_substeps=int(p.get("fast_substeps", 8)),
        transport_correction=p.get("transport_correction", "corrected"),
    )

    # MIT comparison — sync_from_to=True (default) pulls numerics from LOCOMgcm
    # blocks; any override_* (or legacy non-prefixed `viscosity`/`diffusivity`/
    # `dt`/`days`) is honored even under sync mode.
    mc = d.get("mit_comparison", {})
    _reject_unknown_keys(
        "mit_comparison", mc, MitComparisonConfig,
        ("viscosity", "diffusivity", "dt", "days", "tracer_scheme", "multi_dim",
         "mitgcm_adv_scheme", "override_mitgcm_adv_scheme", "viscA4_explicit",
         "override_viscA4_explicit"),
    )

    def _mc_override(name: str) -> Any:
        """Look up override_<name> first, then legacy unprefixed <name>."""
        if f"override_{name}" in mc:
            return mc.get(f"override_{name}")
        if name in mc:
            return mc.get(name)
        return None

    def _mc_override_viscA4() -> Any:
        if "override_viscA4" in mc:
            return mc.get("override_viscA4")
        return _mc_override("viscA4_explicit")

    mit_cmp = MitComparisonConfig(
        enabled=bool(mc.get("enabled", True)),
        sync_from_to=bool(mc.get("sync_from_to", True)),
        viscA4=float(mc.get("viscA4", 0.0)),
        override_viscosity=_mc_override("viscosity"),
        override_diffusivity=_mc_override("diffusivity"),
        override_viscA4=_mc_override_viscA4(),  # avoid clash with legacy 'viscA4'
        override_dt=_mc_override("dt"),
        override_days=_mc_override("days"),
        override_tracer_scheme=_mc_override("tracer_scheme") or _mc_override("mitgcm_adv_scheme"),
        override_multi_dim=_mc_override("multi_dim"),
        #  — MIT-only knobs (no LOCOMgcm counterpart, no sync_from_to dependency).
        cartesian=bool(mc.get("cartesian", False)),
        explicit_viscosity=bool(mc.get("explicit_viscosity", False)),
        sponge_relax_tracers=bool(mc.get("sponge_relax_tracers", False)),
    )

    # Advection
    a = d.get("advection", {})
    _reject_unknown_keys("advection", a, AdvectionConfig)
    adv = AdvectionConfig(
        # absent stays None: AdvectionConfig resolves it to the horizontal scheme
        vertical=a.get("vertical"),
        tracer=a.get("tracer", "dst3_fl"),
        multi_dim=bool(a.get("multi_dim", True)),
    )

    # Boundary
    b = d.get("boundary", {})
    _reject_unknown_keys("boundary", b, BoundaryConfig)
    _opt_float = lambda k: (None if b.get(k) is None else float(b.get(k)))
    boundary = BoundaryConfig(
        type=b.get("type", "open"),
        scheme=b.get("scheme", "clamped"),
        sides=tuple(b.get("sides", ("N", "E", "S", "W"))),
        sponge_width=int(b.get("sponge_width", 3)),
        sponge_mode=b.get("sponge_mode", "mitgcm"),
        sponge_timescale_hours=float(b.get("sponge_timescale_hours", 1.0)),
        sponge_timescale_inner_hours=float(b.get("sponge_timescale_inner_hours", 72.0)),
        sponge_u_timescale_bound_seconds=_opt_float("sponge_u_timescale_bound_seconds"),
        sponge_u_timescale_inner_seconds=_opt_float("sponge_u_timescale_inner_seconds"),
        sponge_v_timescale_bound_seconds=_opt_float("sponge_v_timescale_bound_seconds"),
        sponge_v_timescale_inner_seconds=_opt_float("sponge_v_timescale_inner_seconds"),
        # 
        use_linear_sponge=bool(b.get("use_linear_sponge", False)),
        sponge_ab2_eligible=bool(b.get("sponge_ab2_eligible", False)),
        sponge_balance_obcs_targets=bool(b.get("sponge_balance_obcs_targets", True)),
        boundary_time_mode=b.get("boundary_time_mode", "linear"),
        tangential_zero_mode=b.get("tangential_zero_mode", "none"),
        stevens_relaxation_hours=float(b.get("stevens_relaxation_hours", 6.0)),
        tidal_constituents=tuple(b.get("tidal_constituents", ())),
        tidal_data_dir=b.get("tidal_data_dir", ""),
        tidal_phase_includes_v0=bool(b.get("tidal_phase_includes_v0", False)),
        apply_tangential_velocity=bool(b.get("apply_tangential_velocity", False)),
        obcs_fix_topo=bool(b.get("obcs_fix_topo", False)),
        ob_index_mode=str(b.get("ob_index_mode", "full_edge")),
    )

    # Sources
    sources_raw = d.get("sources", {})
    sources: list[SourceSpec] = []
    if isinstance(sources_raw, dict):
        for name, spec in sources_raw.items():
            if isinstance(spec, dict):
                spec_map = dict(spec)
                stype = spec_map.pop("type", "unknown")
                if stype == "mit_input_binary_mesh":
                    mitcase = dict(case.extras.get("mitcase", {}) or {})
                    spacing = dict(mitcase.get("mesh_spacing", {}) or {})
                    duplicates = sorted(set(spacing) & set(spec_map))
                    if duplicates:
                        raise ValueError(
                            "mit_input_binary_mesh spacing is declared in both "
                            "case.extras.mitcase.mesh_spacing and sources: "
                            + ", ".join(duplicates)
                        )
                    spec_map.update(spacing)
                on_err = spec_map.pop("on_error", "raise")
                cache = spec_map.pop("cache", True)
                tmode = spec_map.pop("time_mode", "sequential")
                sources.append(
                    SourceSpec(
                        name=name, type=stype, config=spec_map,
                        on_error=on_err, cache=cache, time_mode=tmode,
                    )
                )

    # Pipelines (nested: stage_name → list[PipelineSpec])
    pipelines_raw = d.get("pipelines", {})
    pipelines: dict[str, tuple[PipelineSpec, ...]] = {}
    for stage_name, p_list in pipelines_raw.items():
        plist: list[PipelineSpec] = []
        for p_d in p_list or []:
            _reject_unknown_keys(f"pipelines.{stage_name}", p_d, PipelineSpec)
            ops = []
            for op_d in p_d.get("operators", []) or []:
                op_name = op_d.get("op")
                if op_name is None:
                    continue
                op_cfg = {k: v for k, v in op_d.items() if k != "op"}
                ops.append(OperatorSpec(op=op_name, config=op_cfg))
            plist.append(
                PipelineSpec(
                    name=p_d.get("name", ""),
                    primary_source=p_d.get("primary_source", ""),
                    operators=tuple(ops),
                    consumer=p_d.get("consumer", ""),
                    consumer_config=p_d.get("consumer_config", {}),
                )
            )
        pipelines[stage_name] = tuple(plist)

    # Output / Cache / Debug / Compare
    o = d.get("output", {})
    _reject_unknown_keys("output", o, OutputConfig)
    so = o.get("station_output")
    if isinstance(so, dict):
        _reject_unknown_keys("output.station_output", so, StationOutputConfig)
    station_output = None
    if isinstance(so, dict) and so.get("enabled"):
        station_output = StationOutputConfig(
            enabled=True,
            interval_seconds=so.get("interval_seconds"),
            stations=tuple(so.get("stations", ()) or ()),
            variables=tuple(so.get("variables", ("temp", "salt", "u", "v", "eta"))),
        )
    output = OutputConfig(
        root=Path(o.get("root", "outputs/data/")),
        legacy_layout=bool(o.get("legacy_layout", True)),
        monitor_interval_steps=int(o.get("monitor_interval_steps", 10)),
        nc_output_interval_hours=float(o.get("nc_output_interval_hours", 3.0)),
        snapshot_interval_hours=float(o.get("snapshot_interval_hours", 3.0)),
        # Integer-seconds cadence (str expr allowed); wins over hours.
        nc_output_interval_seconds=o.get("nc_output_interval_seconds"),
        snapshot_interval_seconds=o.get("snapshot_interval_seconds"),
        station_output=station_output,
    )

    cmp_d = d.get("compare", {})
    _reject_unknown_keys("compare", cmp_d, CompareConfig)
    compare = CompareConfig(
        variables=tuple(cmp_d.get("variables", ("u", "v", "temperature", "salinity", "eta"))),
        tolerance=cmp_d.get("tolerance", {}),
        output_format=tuple(cmp_d.get("output_format", ("json",))),
    )

    cache_d = d.get("cache", {})
    _reject_unknown_keys("cache", cache_d, CacheConfig)
    cache = CacheConfig(
        root=Path(cache_d.get("root", "outputs/data/_cache/")),
        hycom_cache_dir=Path(cache_d["hycom_cache_dir"]) if cache_d.get("hycom_cache_dir") else None,
        enabled=bool(cache_d.get("enabled", True)),
        cleanup_policy=cache_d.get("cleanup_policy", "after_stage"),
        # 
        rebuild=bool(cache_d.get("rebuild", False)),
        reuse=bool(cache_d.get("reuse", False)),
        verify=bool(cache_d.get("verify", False)),
    )

    dbg = d.get("debug", {})
    _reject_unknown_keys("debug", dbg, DebugConfig)
    debug = DebugConfig(
        persist_intermediate=bool(dbg.get("persist_intermediate", False)),
        log_level=dbg.get("log_level", "INFO"),
        # 
        dump_steps=int(dbg.get("dump_steps", 0)),
    )

    #  — InitialConditionConfig (HYCOM vs WOA23 source selection).
    ic_d = d.get("initial_condition", {}) or {}
    _reject_unknown_keys("initial_condition", ic_d, InitialConditionConfig)
    initial_condition = InitialConditionConfig(
        source=ic_d.get("source", "hycom"),
        woa23_dir=Path(ic_d["woa23_dir"]) if ic_d.get("woa23_dir") else None,
        check_ini_temp=bool(ic_d.get("check_ini_temp", False)),
        allow_freezing=bool(ic_d.get("allow_freezing", False)),
        mit_pickup=(MitPickupConfig(meta=Path(ic_d["mit_pickup"]["meta"]),
                                    data=Path(ic_d["mit_pickup"]["data"]),
                                    n_iter0=int(ic_d["mit_pickup"]["n_iter0"]),
                                    cd_data=(Path(ic_d["mit_pickup"]["cd_data"])
                                             if ic_d["mit_pickup"].get("cd_data") else None))
                    if ic_d.get("mit_pickup") is not None else None),
    )

    #  — AdjointConfig (first-step adjoint trace; surfaced via the
    # separate run_adjoint.py orchestrator entrypoint).
    adj_d = d.get("adjoint", {}) or {}
    _reject_unknown_keys("adjoint", adj_d, AdjointConfig)
    fields_raw = adj_d.get("fields", ("eta", "u", "v", "temp", "salt"))
    adjoint = AdjointConfig(
        target=adj_d.get("target", "hycom"),
        fields=tuple(str(f) for f in fields_raw),
        top_k=int(adj_d.get("top_k", 20)),
        target_npz=Path(adj_d["target_npz"]) if adj_d.get("target_npz") else None,
        detect_anomaly=bool(adj_d.get("detect_anomaly", False)),
    )

    # CheckpointConfig — model-state restart/resume policy.  ``every_steps``
    # / ``every_seconds`` are convenience shortcuts that desugar to a single
    # ``interval`` schedule entry; an explicit ``schedule`` list (if given)
    # is used verbatim and appended to.
    ckpt_d = d.get("checkpoint", {}) or {}
    _reject_unknown_keys("checkpoint", ckpt_d, CheckpointConfig, ("every_steps", "every_seconds"))
    ckpt_schedule = [dict(e) for e in (ckpt_d.get("schedule", []) or [])]
    if ckpt_d.get("every_steps") is not None:
        ckpt_schedule.insert(0, {"type": "interval", "every_steps": int(ckpt_d["every_steps"])})
    if ckpt_d.get("every_seconds") is not None:
        ckpt_schedule.insert(
            0, {"type": "interval", "every_seconds": ckpt_d["every_seconds"]}
        )
    keep_last_raw = ckpt_d.get("keep_last", 3)
    checkpoint = CheckpointConfig(
        enabled=bool(ckpt_d.get("enabled", False)),
        resume=str(ckpt_d.get("resume", "none")),
        resume_dir=(str(ckpt_d["resume_dir"]) if ckpt_d.get("resume_dir") else None),
        schedule=tuple(ckpt_schedule),
        keep_last=None if keep_last_raw is None else int(keep_last_raw),
        permanent_steps=tuple(int(s) for s in (ckpt_d.get("permanent_steps", []) or [])),
        backend=str(ckpt_d.get("backend", "thread")),
        max_inflight=int(ckpt_d.get("max_inflight", 2)),
    )

    observations = None
    observations_d = d.get("observations")
    if observations_d is not None:
        if not isinstance(observations_d, dict):
            raise ValueError("observations must be a mapping")
        _reject_unknown_keys("observations", observations_d, ObservationsConfig)
        section_d = observations_d.get("section_transport_u")
        section = None
        if section_d is not None:
            if not isinstance(section_d, dict):
                raise ValueError("observations.section_transport_u must be a mapping")
            _reject_unknown_keys(
                "observations.section_transport_u", section_d, SectionTransportUConfig
            )
            section = SectionTransportUConfig(
                i_face=section_d.get("i_face"),
                j0=section_d.get("j0"),
                j1=section_d.get("j1"),
            )
        observations = ObservationsConfig(section_transport_u=section)

    # ModelManifest
    m = model_raw
    _reject_unknown_keys("model", m, ModelManifest)
    dynamics_raw = m.get("dynamics", []) or []
    dynamics_entries: list[DynamicsEntry] = []
    for entry in dynamics_raw:
        if isinstance(entry, str):
            # Shorthand: `- linear_free_surface_continuity` is equivalent to
            # `- name: linear_free_surface_continuity`.
            dynamics_entries.append(DynamicsEntry(name=entry, kwargs={}))
        elif isinstance(entry, dict):
            name = entry.get("name")
            if not name:
                raise ValueError(f"model.dynamics entry missing 'name': {entry}")
            _reject_unknown_keys(f"model.dynamics[{name}]", entry, DynamicsEntry)
            kwargs = dict(entry.get("kwargs", {}) or {})
            dynamics_entries.append(DynamicsEntry(name=str(name), kwargs=kwargs))
        else:
            raise ValueError(
                "model.dynamics entry must be str or dict, "
                f"got {type(entry).__name__}"
            )

    stepper_raw = m.get("stepper", {}) or {}
    _reject_unknown_keys("model.stepper", stepper_raw, StepperManifest)
    stepper = StepperManifest(
        name=stepper_raw.get("name"),
        kwargs=dict(stepper_raw.get("kwargs", {}) or {}),
    )

    bc_raw = m.get("boundary_condition", {}) or {}
    _reject_unknown_keys("model.boundary_condition", bc_raw, BoundaryConditionManifest)
    bc = BoundaryConditionManifest(
        name=bc_raw.get("name"),
        kwargs=dict(bc_raw.get("kwargs", {}) or {}),
    )

    model = ModelManifest(
        dynamics=tuple(dynamics_entries),
        stepper=stepper,
        boundary_condition=bc,
    )

    return RunConfig(
        schema_version=schema_version,
        case=case,
        run=run,
        grid=grid,
        time=time_cfg,
        mit_parallel=mit_par,
        physics=physics,
        advection=adv,
        boundary=boundary,
        sources=tuple(sources),
        pipelines=pipelines,
        output=output,
        compare=compare,
        cache=cache,
        debug=debug,
        mit_comparison=mit_cmp,
        initial_condition=initial_condition,
        adjoint=adjoint,
        checkpoint=checkpoint,
        model=model,
        observations=observations,
    )
