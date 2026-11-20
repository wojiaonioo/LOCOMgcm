"""MIT binary consumers.

Binary family:
  - mitgcm_bathy_binary: writes bathyFile.bin (MIT convention: negative ocean)
  - mitgcm_obc_binary: writes per-side OBC binaries
  - mitgcm_exf_binary: writes per-variable surface forcing binaries

Text companions:
  - mitgcm_size_h: writes code/SIZE.h Fortran source
  - mitgcm_data_namelist: writes data namelist text files
  - mitgcm_topology: writes topology yaml + verification log
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from defgcm.io.runner.layer_a import (
    ConventionTag,
    DataBundle,
    LocationKind,
)

from .._mit_binary_base import write_bin_atomic, write_meta_companion
from .._mit_switches import (
    SWITCHES,
    ab_eps_value,
    assert_ab_eps_agrees,
    assert_switches_agree,
    assert_viscosity_agrees,
    assert_windstressmax_agrees,
    bottom_drag_coefficients,
    bottom_drag_module_kwargs,
    eos_namelist_values,
    DataView,
    assert_data_keys_agree,
    cg2d_namelist_values,
    convective_adjustment_frequency,
    gravity_value,
    horizontal_viscosity_module,
    horizontal_viscosity_value,
    leith_smag_coefficients,
    momentum_advection_form,
    momentum_advection_form as momentum_advection_form_of,
    no_slip_sides_value,
    select_ke_scheme_value,
    stepper_field,
    tracer_advection_namelist_values,
    render_switch_lines,
    render_viscosity_lines,
    rho_const_value,
    switch_value,
    tracer_biharmonic_diffusivity,
    tracer_harmonic_diffusivity,
    windstressmax_value,
)
from .._registry import register_consumer
from .._types import (
    ArtifactKind,
    ArtifactRef,
    ConsumerCapabilities,
    ConsumerContext,
)
from ..._time_steps import compute_n_steps, exact_total_seconds
from ...config.vertical_diffusion import bind_mit_vertical_diffusion, resolved_vertical_diffusion
from ...legacy_bridge import TranslationError


@dataclass(frozen=True)
class _MitLegacyExportArgs:
    nx: int
    ny: int
    nz: int
    resolution_deg: float
    center_lat: float
    center_lon: float
    depth_m: float
    layer_thickness: list[float] | None
    dt_seconds: float
    total_seconds: float
    dt_mom_seconds: float | None
    dt_tracer_seconds: float | None
    dt_freesurf_seconds: float | None
    dt_clock_seconds: float | None
    viscosity: float
    viscA4: float
    diffusivity_t: float
    diffusivity_s: float
    eos: dict[str, Any]
    xg_origin: float
    yg_origin: float
    dump_freq_seconds: float
    init_temp: np.ndarray
    init_salt: np.ndarray
    init_u: np.ndarray
    init_v: np.ndarray
    tempAdvScheme: int | None
    saltAdvScheme: int | None
    tempVertAdvScheme: int | None
    saltVertAdvScheme: int | None
    multi_dim_advection: bool
    temperature_implicit_vertical_advection: bool
    salinity_implicit_vertical_advection: bool
    momentum_implicit_vertical_advection: bool
    use_cartesian: bool
    implicit_diffusion: bool
    tracer_vertical_diffusivity_kr: float
    implicit_viscosity: bool
    no_slip_sides: bool
    vertical_viscosity: float
    force_dt: bool
    hfac_min: float
    hfac_min_dz: float
    lin_fs_conserve_tr: bool
    ke_scheme: int
    viscosity_scheme: str
    visc_c2_leith: float
    visc_c2_smag: float
    momentum_advection_form: str
    enable_vertical_shear: bool
    nonhydrostatic: bool
    cg3d_max_iters: int
    cg3d_target_residual: float
    cg3d_target_res_w_unit: float
    nh_am2: float


def _legacy_export_module() -> Any:
    from experiments.alignment import mitgcm_export as legacy

    return legacy


#: ``mit_name -> MitSwitch`` for the handful of call sites that need one row.
_SWITCH_BY_NAME = {s.mit_name: s for s in SWITCHES}


def _bundle_hash(bundle: DataBundle) -> str:
    return bundle.meta.provenance[-1].bundle_hash if bundle.meta.provenance else "00000000"


def _section_transport_enabled(run_cfg: Any) -> bool:
    observations = getattr(run_cfg, "observations", None)
    return getattr(observations, "section_transport_u", None) is not None


def _model_input_dir(ctx: ConsumerContext) -> Path:
    return ctx.output_dir.parent / "_pipeline_meta" / "model_input"


def _read_model_input_run_params(ctx: ConsumerContext) -> dict[str, Any]:
    path = _model_input_dir(ctx) / "run_params.yaml"
    if not path.exists():
        raise FileNotFoundError(f"ModelInput run params not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"ModelInput run params must be a mapping: {path}")
    return data


def _read_model_input_initial_state(ctx: ConsumerContext) -> dict[str, np.ndarray]:
    from defgcm.io.runner.model_input import read_initial_state_dataset

    path = _model_input_dir(ctx) / "initial_state.zarr"
    if not path.exists():
        raise FileNotFoundError(f"ModelInput initial state not found: {path}")
    ds = read_initial_state_dataset(path)
    try:
        return {
            name: np.asarray(ds[name].values, dtype=np.float64)
            for name in ("eta", "temp", "salt", "u", "v")
        }
    finally:
        ds.close()


def _read_model_input_grid(ctx: ConsumerContext) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    grid_dir = _model_input_dir(ctx) / "grid"
    scalars_path = grid_dir / "scalars.yaml"
    tensors_path = grid_dir / "tensors.npz"
    if not scalars_path.exists():
        raise FileNotFoundError(f"ModelInput grid scalars not found: {scalars_path}")
    if not tensors_path.exists():
        raise FileNotFoundError(f"ModelInput grid tensors not found: {tensors_path}")
    scalars = yaml.safe_load(scalars_path.read_text()) or {}
    with np.load(tensors_path) as npz:
        tensors = {name: np.asarray(npz[name]) for name in npz.files}
    return scalars, tensors


def _run_param(params: dict[str, Any], name: str, default: Any = None) -> Any:
    return params[name] if params.get(name) is not None else default


def _required_implicit_vertical_advection_flags(
    params: dict[str, Any],
) -> tuple[bool, bool, bool]:
    fields = (
        ("temperature", "temperature_implicit_vertical_advection"),
        ("salinity", "salinity_implicit_vertical_advection"),
        ("momentum", "momentum_implicit_vertical_advection"),
    )
    nested = params.get("implicit_vertical_advection")
    duplicate_paths: list[str] = []
    if isinstance(nested, dict):
        for leaf, flat_key in fields:
            if leaf in nested and flat_key in params:
                duplicate_paths.extend(
                    [
                        f"ModelInput.run_params.implicit_vertical_advection.{leaf}",
                        f"ModelInput.run_params.{flat_key}",
                    ]
                )
    if duplicate_paths:
        raise ValueError(
            "implicit vertical advection has duplicate ModelInput source(s): "
            + ", ".join(duplicate_paths)
        )

    resolved: list[bool] = []
    for _, flat_key in fields:
        if flat_key not in params:
            raise ValueError(f"ModelInput.run_params.{flat_key} is required")
        value = params[flat_key]
        if type(value) is not bool:
            raise ValueError(
                f"ModelInput.run_params.{flat_key} must be bool, got "
                f"{type(value).__name__}"
            )
        resolved.append(value)
    return resolved[0], resolved[1], resolved[2]


def _mit_vertical_advection_scheme(name: Any, *, implicit_advection: bool) -> int:
    """MITgcm ``tempVertAdvScheme`` of a vertical scheme name, from the scheme declarations.

    The scheme must be in ``VERTICAL_TRACER_SCHEMES`` (explicit) or
    ``IMPLICIT_VERTICAL_TRACER_SCHEMES`` (implicit column); the code is its
    ``TRACER_ADVECTION_SCHEMES`` entry.
    """
    from defgcm.dynamics.implicit_vertical_advection import IMPLICIT_VERTICAL_TRACER_SCHEMES
    from defgcm.operators.fluxes.tracer_flux import (
        TRACER_SCHEME_BY_NAME,
        VERTICAL_TRACER_SCHEMES,
    )

    scheme = str(name or "upwind").lower()
    allowed = IMPLICIT_VERTICAL_TRACER_SCHEMES if implicit_advection else VERTICAL_TRACER_SCHEMES
    if scheme not in allowed:
        raise ValueError(
            f"unsupported {'implicit' if implicit_advection else 'explicit'} vertical advection "
            f"scheme for MITgcm export: {name!r}; declared: {', '.join(allowed)}"
        )
    return TRACER_SCHEME_BY_NAME[scheme].mit_code


def _legacy_nh_params(params: dict[str, Any]) -> Any:
    from defgcm.dycore_parameters import NonhydrostaticParams

    return NonhydrostaticParams(
        nonhydrostatic=bool(_run_param(params, "nonhydrostatic", False)),
        cg3d_max_iters=int(_run_param(params, "cg3d_max_iters", 150)),
        cg3d_target_residual=float(_run_param(params, "cg3d_target_residual", 1e-7)),
        cg3d_target_res_w_unit=float(_run_param(params, "cg3d_target_res_w_unit", -1.0)),
        nh_am2=float(_run_param(params, "nh_am2", 1.0)),
    )


def vertical_diffusion_export_args(run_cfg):
    """Assign the legacy writer's two arguments with MIT binding receipts."""
    kwargs = {}
    def write(key, value):
        field = ("implicit_diffusion" if key == "implicitDiffusion"
                 else "tracer_vertical_diffusivity_kr")
        kwargs[field] = value

    _, bindings = bind_mit_vertical_diffusion(
        resolved_vertical_diffusion(run_cfg, error_type=TranslationError), write=write,
    )
    return kwargs, bindings


def _assemble_legacy_export_args(ctx: ConsumerContext) -> _MitLegacyExportArgs:
    if ctx.run_cfg is None:
        raise TranslationError("MIT tracer diffusion export requires RunConfig")
    tracer_diffusion, _ = vertical_diffusion_export_args(ctx.run_cfg)
    params = _read_model_input_run_params(ctx)
    if bool(_run_param(params, "mitgcm_cartesian", False)):
        # The writer's Cartesian branch needs the grid spacing and the Coriolis
        # f0, beta, which this consumer does not carry.
        raise TranslationError(
            "mit_comparison.cartesian: the data consumer exports spherical-polar "
            "grids only; it does not carry the Cartesian delX/delY and Coriolis f0, beta"
        )
    (
        temperature_implicit_vertical_advection,
        salinity_implicit_vertical_advection,
        momentum_implicit_vertical_advection,
    ) = _required_implicit_vertical_advection_flags(params)
    _read_model_input_grid(ctx)
    state = _read_model_input_initial_state(ctx)
    legacy = _legacy_export_module()

    nx = int(_run_param(params, "nx"))
    ny = int(_run_param(params, "ny"))
    nz = int(_run_param(params, "nz"))
    resolution_deg = float(_run_param(params, "resolution_deg"))
    center_lat = float(_run_param(params, "center_lat"))
    center_lon = float(_run_param(params, "center_lon"))
    depth_m = float(_run_param(params, "depth_m"))
    layer_thickness = _run_param(params, "vertical_layer_thicknesses")
    if _run_param(params, "vertical_center_spacing"):
        raise ValueError(
            "grid.vertical_center_spacing (MITgcm delRc) is not exported: the MITgcm "
            "namelist writer emits delZ only, which would put the centres midway"
        )
    if layer_thickness is not None:
        layer_thickness = [float(v) for v in layer_thickness]

    projected = legacy.project_state_to_mitgcm_initial_fields(
        type("_ModelInputState", (), {
            "eta": type("_Field", (), {"data": state["eta"]})(),
            "temp": type("_Field", (), {"data": state["temp"]})(),
            "salt": type("_Field", (), {"data": state["salt"]})(),
            "u": type("_Field", (), {"data": state["u"]})(),
            "v": type("_Field", (), {"data": state["v"]})(),
        })(),
        nx=nx,
        ny=ny,
        nz=nz,
    )

    run_cfg = ctx.run_cfg
    if run_cfg is not None:
        yaml_days = run_cfg.time.days
        yaml_steps_override = run_cfg.time.steps
        yaml_seconds_override = run_cfg.time.seconds
        dt_seconds = float(run_cfg.time.dt)
        dt_mom_seconds = float(run_cfg.time.dt_mom)
        dt_tracer_seconds = float(run_cfg.time.dt_tracer)
        dt_freesurf_seconds = float(run_cfg.time.dt_freesurf)
        dt_clock_seconds = float(run_cfg.time.dt_clock)
        nc_output_interval_seconds = run_cfg.output.resolved_nc_output_seconds()
    else:
        yaml_days = _run_param(params, "days", None)
        yaml_steps_override = _run_param(params, "steps", None)
        yaml_seconds_override = _run_param(params, "seconds", None)
        dt_seconds = float(_run_param(params, "fixed_dt", _run_param(params, "dt")))
        dt_mom_seconds = params.get("dt_mom")
        dt_tracer_seconds = params.get("dt_tracer")
        dt_freesurf_seconds = params.get("dt_freesurf")
        dt_clock_seconds = params.get("dt_clock")
        nc_output_interval_seconds = _run_param(params, "nc_output_interval_seconds", None)
    if dt_seconds <= 0.0:
        raise ValueError(f"resolved MITgcm dt must be positive, got {dt_seconds}")

    max_u = float(np.abs(projected["u"]).max())
    max_v = float(np.abs(projected["v"]).max())
    max_speed = max(max_u, max_v, 0.1)
    force_dt = any(
        params.get(name) is not None
        for name in ("fixed_dt", "dt", "dt_mom", "dt_tracer", "dt_freesurf", "dt_clock")
    )
    if not force_dt:
        dx_m = resolution_deg * 111_320 * math.cos(math.radians(center_lat))
        dy_m = resolution_deg * 110_574
        cfl_dt_h = min(dx_m, dy_m) / max_speed * 0.3
        dt_seconds = min(dt_seconds, cfl_dt_h, 120.0)
        dt_seconds = max(dt_seconds, 1.0)

    mitgcm_dt_mom = dt_seconds if dt_mom_seconds is None else float(dt_mom_seconds)
    mitgcm_dt_tracer = dt_seconds if dt_tracer_seconds is None else float(dt_tracer_seconds)
    mitgcm_dt_freesurf = dt_seconds if dt_freesurf_seconds is None else float(dt_freesurf_seconds)
    mitgcm_dt_clock = dt_seconds if dt_clock_seconds is None else float(dt_clock_seconds)
    # Step count routes through the SSOT helper. The same function feeds the
    # LOCOMgcm runtime (experiments/cases/_runtime.py:_steps_from_config), so MIT
    # and LOCOMgcm are guaranteed to agree on n_steps. endTime is then derived
    # from n_steps * dt (integer arithmetic), not from days * 86400, so it
    # is always an exact integer multiple of dt.
    n_steps = compute_n_steps(
        days=None if yaml_days is None else float(yaml_days),
        dt_seconds=mitgcm_dt_clock,
        steps_override=None if yaml_steps_override is None else int(yaml_steps_override),
        seconds_override=None if yaml_seconds_override is None else float(yaml_seconds_override),
        context="mit_binary._assemble_legacy_export_args",
    )
    adjusted_total = exact_total_seconds(n_steps, mitgcm_dt_clock)

    # Dump cadence: prefer the integer-seconds field (avoids float-hours
    # like 0.00416667 for "every step"); fall back to legacy hours.
    if nc_output_interval_seconds is not None:
        dump_freq_seconds = float(nc_output_interval_seconds)
    else:
        dump_freq_seconds = float(_run_param(params, "nc_output_interval_hours", 3.0)) * 3600.0
    if dump_freq_seconds > 0:
        interval_steps = max(1, int(round(dump_freq_seconds / mitgcm_dt_clock)))
        dump_freq_seconds = interval_steps * mitgcm_dt_clock

    # viscAh, diffKhT and diffKhS as LOCOMgcm's modules run them, unless
    # mit_comparison overrides them on purpose (MitComparisonConfig).
    comparison = run_cfg.mit_comparison
    viscosity = comparison.effective_viscosity(horizontal_viscosity_value(run_cfg))
    diffusivity_t = comparison.effective_diffusivity(tracer_harmonic_diffusivity(run_cfg, "temp"))
    diffusivity_s = comparison.effective_diffusivity(tracer_harmonic_diffusivity(run_cfg, "salt"))
    # LINEAR tRef/sRef: a module without its own takes the runtime profiles,
    # resolved by the function enrich_manifest calls, here from the ModelInput
    # initial state.
    from ..._canonical_helpers import resolve_reference_profiles
    import torch

    advection = tracer_advection_namelist_values(run_cfg)
    viscosity_module = horizontal_viscosity_module(run_cfg)
    viscosity_scheme = "constant" if viscosity_module is None else str(viscosity_module.viscosity_scheme)
    visc_c2_leith, visc_c2_smag = leith_smag_coefficients(run_cfg)
    eos = eos_namelist_values(
        run_cfg,
        reference_profiles=resolve_reference_profiles(
            run_cfg,
            None,
            {"temperature": torch.as_tensor(state["temp"]),
             "salinity": torch.as_tensor(state["salt"])},
            device="cpu",
        ),
    )
    hfac_min_dz = params.get("hfac_min_dz")
    if hfac_min_dz is None:
        layer_for_hfac = np.asarray(layer_thickness, dtype=np.float64) if layer_thickness else None
        min_dz = float(layer_for_hfac.min()) if layer_for_hfac is not None else depth_m / nz
        hfac_min_dz = min(min_dz * float(legacy.MITGCM_HFAC_MIN), float(legacy.MITGCM_HFAC_MIN_DZ))

    (
        nonhydrostatic_enabled,
        cg3d_max_iters,
        cg3d_target_residual,
        cg3d_target_res_w_unit,
        nh_am2,
    ) = legacy._resolve_nonhydrostatic_export_controls(
        _legacy_nh_params(params),
        None,
        None,
    )

    return _MitLegacyExportArgs(
        nx=nx,
        ny=ny,
        nz=nz,
        resolution_deg=resolution_deg,
        center_lat=center_lat,
        center_lon=center_lon,
        depth_m=depth_m,
        layer_thickness=layer_thickness,
        dt_seconds=dt_seconds,
        total_seconds=adjusted_total,
        dt_mom_seconds=mitgcm_dt_mom,
        dt_tracer_seconds=mitgcm_dt_tracer,
        dt_freesurf_seconds=mitgcm_dt_freesurf,
        dt_clock_seconds=mitgcm_dt_clock,
        viscosity=viscosity,
        viscA4=abs(float(_run_param(params, "mitgcm_viscA4", 0.0) or 0.0)),
        diffusivity_t=diffusivity_t,
        diffusivity_s=diffusivity_s,
        eos=eos,
        xg_origin=center_lon - (nx / 2) * resolution_deg,
        yg_origin=center_lat - (ny / 2) * resolution_deg,
        dump_freq_seconds=dump_freq_seconds,
        init_temp=projected["temp"],
        init_salt=projected["salt"],
        init_u=projected["u"],
        init_v=projected["v"],
        # Each tracer's advecting module (layer_d/_mit_switches.py
        # tracer_advection_namelist_values), not global advection fields or a
        # fixed multiDimAdvection=.TRUE.
        tempAdvScheme=advection.get("tempAdvScheme"),
        saltAdvScheme=advection.get("saltAdvScheme"),
        tempVertAdvScheme=advection.get("tempVertAdvScheme"),
        saltVertAdvScheme=advection.get("saltVertAdvScheme"),
        multi_dim_advection=bool(advection["multiDimAdvection"]),
        temperature_implicit_vertical_advection=(
            temperature_implicit_vertical_advection
        ),
        salinity_implicit_vertical_advection=salinity_implicit_vertical_advection,
        momentum_implicit_vertical_advection=momentum_implicit_vertical_advection,
        use_cartesian=bool(_run_param(params, "mitgcm_cartesian", False)),
        implicit_diffusion=tracer_diffusion["implicit_diffusion"],
        tracer_vertical_diffusivity_kr=tracer_diffusion["tracer_vertical_diffusivity_kr"],
        implicit_viscosity=not bool(_run_param(params, "mitgcm_explicit_viscosity", False)),
        # no_slip_sides + viscAz (vertical viscosity) reach the MIT export from
        # the SAME canonical source LOCOMgcm reads (model.dynamics.horizontal_viscosity
        # / stepper.implicit_viscosity via legacy_bridge) so the two models can
        # not silently diverge. The no_slip_sides fallback (free-slip) is
        # checked by assert_viscosity_agrees; viscAz is the resolved value,
        # 0 included (legacy_bridge._implicit_viscosity always emits it).
        no_slip_sides=no_slip_sides_value(run_cfg),
        vertical_viscosity=float(params["implicit_viscosity"]),
        force_dt=force_dt,
        # Read hfac_min from run_params (yaml's case.extras.hfac_min, via
        # legacy_bridge). A hardcoded MITGCM_HFAC_MIN=0.3 against LOCOMgcm's
        # default 0.2 makes the MIT step 0 dump zero out wet-but-thin cells
        # (max|T| diff = 29K).
        hfac_min=float(_run_param(params, "hfac_min", legacy.MITGCM_HFAC_MIN)),
        hfac_min_dz=float(hfac_min_dz),
        lin_fs_conserve_tr=bool(_run_param(params, "lin_fs_conserve_tr", False)),
        ke_scheme=select_ke_scheme_value(run_cfg),
        viscosity_scheme=viscosity_scheme,
        visc_c2_leith=visc_c2_leith,
        visc_c2_smag=visc_c2_smag,
        momentum_advection_form=momentum_advection_form(run_cfg),
        enable_vertical_shear=bool(_run_param(params, "enable_vertical_shear", True)),
        nonhydrostatic=nonhydrostatic_enabled,
        cg3d_max_iters=cg3d_max_iters,
        cg3d_target_residual=cg3d_target_residual,
        cg3d_target_res_w_unit=cg3d_target_res_w_unit,
        nh_am2=nh_am2,
    )


def _legacy_start_date_ymd(ctx: ConsumerContext) -> int:
    params = _read_model_input_run_params(ctx)
    raw = str(_run_param(params, "start_date", "2019-01-01"))
    return int(raw.replace("-", ""))


def _legacy_sponge_width(ctx: ConsumerContext) -> int:
    params = _read_model_input_run_params(ctx)
    return int(_run_param(params, "mitgcm_sponge_width", _run_param(params, "sponge_width", 0)))


def _extract_obcs_timing(bundle: DataBundle) -> tuple[float, float, int]:
    """Read the OBC time axis from a boundary bundle.

    Returns (obcs_period_seconds, repeat_period_seconds, n_records) matching
    legacy mitgcm_export.py:
      obcs_period        = times[1] - times[0]
      obcs_repeat_period = times[-1] - times[0]   (legacy uses times[-1] with
                           a 0-based time axis, i.e. span from t0)

    A SINGLE time record (n==1) is a STEADY boundary (held constant) — e.g.
    Gong ISWFM, whose boundary T/S come from one IC snapshot and never vary in
    time (only tides do). Returns period 0.0 (= MITgcm OBCS_monitorFreq=0,
    monitoring off; the prescribe field is read once and held). This needs no
    EXF time-interpolation, matching Gong's own data.obcs which runs
    useOBCSprescribe with no time-varying files and no EXF.
    """
    if "time" not in bundle.data.coords:
        raise ValueError("_extract_obcs_timing: boundary bundle has no 'time' coord")
    t = np.asarray(bundle.data["time"].values)
    if t.size == 0:
        raise ValueError("_extract_obcs_timing: boundary bundle has 0 OBC time records")
    if t.size == 1:
        return 0.0, 0.0, 1
    if np.issubdtype(t.dtype, np.datetime64):
        secs = (t - t[0]) / np.timedelta64(1, "s")
        secs = np.asarray(secs, dtype=np.float64)
    else:
        secs = np.asarray(t, dtype=np.float64)
        secs = secs - secs[0]
    obcs_period = float(secs[1] - secs[0])
    repeat_period = float(secs[-1])
    return obcs_period, repeat_period, int(t.size)


@register_consumer("mitgcm_ic_binary")
class MitgcmIcBinaryAdapter:
    """Serialize ModelInput.initial_state to MITgcm IC binaries.

    The input bundle is already model-ready: temp/salt are masked, u/v are
    C-grid face velocities, and OBC strips have already been enforced by
    ``build_model_state``.  This adapter only drops MITgcm's unused east/north
    faces, replaces NaNs with zero, and writes big-endian float64 files.
    """

    capabilities = ConsumerCapabilities(
        accepts_location_kinds=(LocationKind.GRIDDED_3D_OCEAN,),
        required_convention=ConventionTag.TORCH_OCEAN,
        required_variables=("temp", "salt", "u", "v"),
        artifact_kind=ArtifactKind.FILE_BINARY,
        artifact_schema_version="0.1.0",
    )

    # `eta` is exported when present in the bundle; missing it caused a 1.2 m
    # SSH offset between LOCOMgcm and MIT (MIT's INI_PSURF defaults etaN=0 when
    # pSurfInitFile is unset).  See PARM05 pSurfInitFile in data namelist.
    _VAR_TO_FILE = {
        "temp": "hydrogThetaFile.bin",
        "salt": "hydrogSaltFile.bin",
        "u": "uVelInitFile.bin",
        "v": "vVelInitFile.bin",
        "eta": "pSurfInitFile.bin",
    }

    def __init__(self, ctx: ConsumerContext, *, dtype: str = "float64", **_: Any) -> None:
        self._ctx = ctx
        self._dtype = dtype

    def consume(self, bundle: DataBundle) -> ArtifactRef:
        if bundle.meta.convention != ConventionTag.TORCH_OCEAN:
            raise ValueError(
                f"mitgcm_ic_binary: requires TORCH_OCEAN convention, got {bundle.meta.convention}"
            )
        if self._ctx.run_cfg is None:
            raise ValueError("mitgcm_ic_binary: ctx.run_cfg is None")

        input_dir = self._ctx.output_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        g = self._ctx.run_cfg.grid
        nx, ny, nz = int(g.nx), int(g.ny), int(g.nz)

        def _state_array(var: str) -> np.ndarray:
            arr = np.asarray(bundle.data[var].values, dtype=np.float64)
            if arr.ndim == 4:
                if arr.shape[0] != 1:
                    raise ValueError(f"mitgcm_ic_binary: {var} has multiple time/batch records {arr.shape}")
                arr = arr[0]
            return arr

        def _project(var: str) -> np.ndarray:
            arr = _state_array(var)
            if var == "temp" or var == "salt":
                expected = (nz, ny, nx)
            elif var == "u":
                expected = (nz, ny, nx + 1)
            elif var == "v":
                expected = (nz, ny + 1, nx)
            elif var == "eta":
                # eta is 2-D surface field; accept (ny, nx) or (1, ny, nx)
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]
                expected = (ny, nx)
            else:  # pragma: no cover - _VAR_TO_FILE is closed
                raise ValueError(f"unknown IC variable {var!r}")
            if arr.shape != expected:
                raise ValueError(
                    f"mitgcm_ic_binary: {var} shape {arr.shape} != expected {expected}"
                )
            if var == "u":
                arr = arr[..., :nx]
            elif var == "v":
                arr = arr[..., :ny, :]
            return np.ascontiguousarray(np.nan_to_num(arr, nan=0.0).astype(np.float64))

        out_files: list[tuple[str, str]] = []
        for var, fname in self._VAR_TO_FILE.items():
            arr = _project(var)
            fpath = input_dir / fname
            write_bin_atomic(arr, fpath, dtype=self._dtype, endianness="big")
            write_meta_companion(
                fpath.with_suffix(".meta"),
                dim_list=[(nx, 1, nx), (ny, 1, ny), (nz, 1, nz)],
                dataprec=self._dtype,
                nrecords=1,
                time_step_number=0,
            )
            out_files.append((var, fname))

        if not out_files:
            raise ValueError("mitgcm_ic_binary: no IC variables found in bundle")

        # Index: one ``<var>\t<file>`` line per binary written into input_dir.
        primary = input_dir / "ic_index.txt"
        primary.write_text("\n".join(f"{v}\t{f}" for v, f in out_files) + "\n")
        return ArtifactRef(
            name="mitgcm_ic_binary",
            path=primary,
            artifact_kind=ArtifactKind.FILE_BINARY,
            bundle_hash=bundle.meta.provenance[-1].bundle_hash if bundle.meta.provenance else "00000000",
            schema_version="0.1.0",
            extras={
                "input_dir": str(input_dir),
                "vars": [v for v, _ in out_files],
                "shape": [nz, ny, nx],
                "dtype": self._dtype,
                "endianness": "big",
            },
        )


@register_consumer("mitgcm_bathy_binary")
class MitgcmBathyAdapter:
    """Writes MIT bathyFile.bin (negative-ocean convention, big-endian float64).

    Input bundle MUST be MITGCM convention.  The ConventionConverter upstream
    (``convention_convert`` to ``mitgcm``) produces the negative-ocean values
    through ``experiments/alignment/mitgcm_export.py``
    ``project_bathymetry_to_mitgcm`` for LOCOMgcm depth and for GEBCO
    elevation alike; this adapter writes them unchanged.
    """

    capabilities = ConsumerCapabilities(
        accepts_location_kinds=(LocationKind.MESH,),
        required_convention=ConventionTag.MITGCM,
        required_variables=("bathymetry",),
        artifact_kind=ArtifactKind.FILE_BINARY,
        artifact_schema_version="0.1.0",
    )

    def __init__(self, ctx: ConsumerContext, *, dtype: str = "float64", **_: Any) -> None:
        self._ctx = ctx
        self._dtype = dtype

    def consume(self, bundle: DataBundle) -> ArtifactRef:
        self._pre_validate(bundle)
        bathy = np.asarray(bundle.data["bathymetry"].values)
        # bathymetry should be 2D (y, x); MITgcm reads (nx, ny) Fortran-flat,
        # which equals (ny, nx) C-row-major when written in C order.
        if bathy.ndim != 2:
            raise ValueError(
                f"mitgcm_bathy_binary: bathymetry must be 2D, got shape {bathy.shape}"
            )
        ny, nx = bathy.shape

        input_dir = self._ctx.output_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        final_path = input_dir / "bathyFile.bin"
        write_bin_atomic(bathy, final_path, dtype=self._dtype, endianness="big")

        # .meta companion: 2 dims, full domain
        meta_path = input_dir / "bathyFile.meta"
        write_meta_companion(
            meta_path,
            # Format: (global_size, local_start_1based, local_end_1based) per dim.
            # Full-domain single-tile bathy: local_start=1, local_end=global_size.
            dim_list=[(nx, 1, nx), (ny, 1, ny)],
            dataprec=self._dtype,
            nrecords=1,
            time_step_number=0,
        )

        return ArtifactRef(
            name="mitgcm_bathy_binary",
            path=final_path,
            artifact_kind=ArtifactKind.FILE_BINARY,
            bundle_hash=bundle.meta.provenance[-1].bundle_hash if bundle.meta.provenance else "00000000",
            schema_version="0.1.0",
            extras={
                "size_bytes": final_path.stat().st_size,
                "endianness": "big",
                "meta_path": str(meta_path),
            },
        )

    def _pre_validate(self, bundle: DataBundle) -> None:
        if bundle.meta.location_kind != LocationKind.MESH:
            raise ValueError(
                f"mitgcm_bathy_binary: requires MESH bundle; got {bundle.meta.location_kind}"
            )
        if bundle.meta.convention != ConventionTag.MITGCM:
            raise ValueError(
                f"mitgcm_bathy_binary: requires MITGCM convention; got {bundle.meta.convention}"
            )
        if "bathymetry" not in bundle.data.data_vars:
            raise ValueError("mitgcm_bathy_binary: bundle missing 'bathymetry' variable")


# ============================================================
# mitgcm_obc_binary (Profile S)
# ============================================================


@register_consumer("mitgcm_obc_binary")
class MitgcmObcAdapter:
    """Writes per-side OBC binaries (e.g. obc_T_N.bin, obc_T_E.bin, ...).

    For each requested variable + side, writes a binary file with the
    boundary-along time series in MIT convention.
    """

    capabilities = ConsumerCapabilities(
        accepts_location_kinds=(LocationKind.BOUNDARY_TIMESERIES,),
        required_convention=ConventionTag.MITGCM,
        required_variables=(),  # caller decides which vars
        artifact_kind=ArtifactKind.FILE_BINARY,
        artifact_schema_version="0.1.0",
    )

    def __init__(self, ctx: ConsumerContext, *, dtype: str = "float64", **_: Any) -> None:
        self._ctx = ctx
        self._dtype = dtype

    # Canonical var name → MITgcm OBCS file suffix (matches legacy
    # mitgcm_export.py `_MITGCM_OBCS_FIELD_SPECS`: temp→t, salt→s, u→u, v→v).
    _VAR_SUFFIX = {
        "temperature": "t",
        "salinity": "s",
        "u": "u",
        "v": "v",
    }

    def consume(self, bundle: DataBundle) -> ArtifactRef:
        self._pre_validate(bundle)
        out_files: list[tuple[str, str, str]] = []

        sides_coord = bundle.data["side"].values if "side" in bundle.data.coords else ["N", "E", "S", "W"]
        # Non-square domains zero-pad `along` to max(nx, ny); slice each side to
        # its true length (N/S=nx, E/W=ny) so the OBC bins match MIT's per-side
        # face length. Square domains have no along_valid (all == nx) → no-op.
        along_valid = (
            np.asarray(bundle.data["along_valid"].values, dtype=int)
            if "along_valid" in bundle.data.coords else None
        )

        input_dir = self._ctx.output_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        for var in bundle.meta.variables:
            suffix = self._VAR_SUFFIX.get(var.name)
            if suffix is None:
                continue
            da = bundle.data[var.name]
            arr = np.asarray(da.values).astype(np.float64)
            # HYCOM boundary strips carry land/missing NaN — fill before
            # write (MITgcm OBCS prescribe propagates NaN into the interior).
            # u/v → 0; tracers → nanmean (land cells masked by hFac anyway).
            if np.isnan(arr).any():
                if var.name in ("u", "v"):
                    arr = np.nan_to_num(arr, nan=0.0)
                else:
                    fill = np.nanmean(arr) if not np.isnan(arr).all() else 0.0
                    arr = np.where(np.isnan(arr), fill, arr)
            # Shape: (time, side, [z], along)
            n_side = arr.shape[1]
            for i in range(n_side):
                side_name = str(sides_coord[i]) if i < len(sides_coord) else f"S{i}"
                side_arr = arr[:, i, ...]  # (time, z?, along) or (time, along)
                if along_valid is not None:
                    side_arr = side_arr[..., : int(along_valid[i])]  # drop pad tail
                # Legacy naming: OB{Side}{suffix}.bin (e.g. OBNt.bin, OBEu.bin).
                fname = f"OB{side_name}{suffix}.bin"
                fpath = input_dir / fname
                write_bin_atomic(side_arr, fpath, dtype=self._dtype, endianness="big")
                # .meta sidecar: dimList in Fortran order
                dim_list_meta = [(s, 1, s) for s in side_arr.shape]
                write_meta_companion(
                    fpath.with_suffix(".meta"),
                    dim_list=dim_list_meta,
                    dataprec=self._dtype,
                    nrecords=side_arr.shape[0],
                )
                out_files.append((var.name, side_name, fname))

        # Index: one ``<var>\t<side>\t<file>`` line per binary written into input_dir.
        primary = input_dir / "obc_index.txt"
        index_lines = [f"{var}\t{side}\t{f}" for (var, side, f) in out_files]
        primary.write_text("\n".join(index_lines) + "\n")

        return ArtifactRef(
            name="mitgcm_obc_binary",
            path=primary,
            artifact_kind=ArtifactKind.FILE_BINARY,
            bundle_hash=bundle.meta.provenance[-1].bundle_hash if bundle.meta.provenance else "00000000",
            schema_version="0.1.0",
            extras={
                "n_files": len(out_files),
                "files": [(v, s) for (v, s, _) in out_files],
            },
        )

    def _pre_validate(self, bundle: DataBundle) -> None:
        if bundle.meta.location_kind != LocationKind.BOUNDARY_TIMESERIES:
            raise ValueError(f"mitgcm_obc_binary: requires BOUNDARY_TIMESERIES; got {bundle.meta.location_kind}")
        if bundle.meta.convention != ConventionTag.MITGCM:
            raise ValueError(f"mitgcm_obc_binary: requires MITGCM; got {bundle.meta.convention}")


# ============================================================
# mitgcm_tidal_binary (Profile S)
# ============================================================


@register_consumer("mitgcm_tidal_binary")
class MitgcmTidalBinaryAdapter:
    """Writes per-constituent tidal OBCS binaries: OB{side}_{u,v}Tid{Am,Ph}_{c}.bin.

    Consumes the canonical TIDAL_HARMONICS bundle (Greenwich amplitude m/s +
    phase seconds, dims (constituent, side, along)) and writes one little-endian
    float64 file per (side, component, amp/phase, constituent) — byte-for-byte
    identical to the verified extract_tpxo9_tides.py output. These per-constituent
    Greenwich files are the intermediate that mitgcm_obcs_namelist later stacks +
    V0-corrects + flips to big-endian for the MITgcm OBCS tidal namelist.

    Unlike the other mitgcm_* binary consumers this accepts CF_STANDARD (not
    MITGCM): tidal amplitude is a magnitude and phase is a time offset — neither
    carries the LOCOMgcm↔MIT sign flip, so the tidal lane needs no convention_convert.

    Endianness: LITTLE-endian (matches extract_tpxo9_tides.py `arr.tofile()` and
    the way mitgcm_obcs_namelist reads them back via `np.fromfile(dtype=float64)`).
    The big-endian conversion happens only on the stacked MIT-input files.
    """

    capabilities = ConsumerCapabilities(
        accepts_location_kinds=(LocationKind.TIDAL_HARMONICS,),
        required_convention=ConventionTag.CF_STANDARD,
        required_variables=(),
        artifact_kind=ArtifactKind.FILE_BINARY,
        artifact_schema_version="0.1.0",
    )

    # bundle var name → MITgcm tidal file prefix.
    _VAR_PREFIX = {
        "u_tide_amplitude": "uTidAm",
        "u_tide_phase": "uTidPh",
        "v_tide_amplitude": "vTidAm",
        "v_tide_phase": "vTidPh",
    }

    # Output subdir under <run>/mitgcm/input — read by mitgcm_obcs_namelist.
    SUBDIR = "tides_raw"

    def __init__(self, ctx: ConsumerContext, **_: Any) -> None:
        self._ctx = ctx

    def consume(self, bundle: DataBundle) -> ArtifactRef:
        if bundle.meta.location_kind != LocationKind.TIDAL_HARMONICS:
            raise ValueError(
                f"mitgcm_tidal_binary: requires TIDAL_HARMONICS; got {bundle.meta.location_kind}"
            )
        ds = bundle.data
        constituents = [str(c) for c in ds["constituent"].values]
        sides = [str(s) for s in ds["side"].values]
        # Each side's true length (N/S=nx, E/W=ny). For non-square domains the
        # `along` axis is zero-padded to max(nx, ny); write only the valid prefix
        # so the bins match the script's per-side length exactly.
        valid = (
            np.asarray(ds["along_valid"].values, dtype=int)
            if "along_valid" in ds.coords else None
        )

        out_dir = self._ctx.output_dir / "input" / self.SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)

        out_files: list[str] = []
        for var in bundle.meta.variables:
            pfx = self._VAR_PREFIX.get(var.name)
            if pfx is None:
                continue
            arr = np.asarray(ds[var.name].values, dtype="float64")  # (const, side, along)
            for c_idx, cname in enumerate(constituents):
                for s_idx, side in enumerate(sides):
                    vlen = int(valid[s_idx]) if valid is not None else arr.shape[-1]
                    line = np.ascontiguousarray(arr[c_idx, s_idx, :vlen])  # (vlen,)
                    fname = f"OB{side}_{pfx}_{cname}.bin"
                    fpath = out_dir / fname
                    # LITTLE-endian float64 — bitwise identical to the script's
                    # `arr.astype(np.float64).tofile()`.
                    write_bin_atomic(line, fpath, dtype="float64", endianness="little")
                    out_files.append(fname)

        # Index: one file name per line, for every raw file written into out_dir.
        index = out_dir / "tidal_index.txt"
        index.write_text("\n".join(out_files) + "\n")

        return ArtifactRef(
            name="mitgcm_tidal_binary",
            path=out_dir,
            artifact_kind=ArtifactKind.FILE_BINARY,
            bundle_hash=bundle.meta.provenance[-1].bundle_hash if bundle.meta.provenance else "00000000",
            schema_version="0.1.0",
            extras={
                "n_files": len(out_files),
                "constituents": tuple(constituents),
                "sides": tuple(sides),
            },
        )


# ============================================================
# mitgcm_exf_binary (Profile S)
# ============================================================


@register_consumer("mitgcm_exf_binary")
class MitgcmExfAdapter:
    """Writes MIT EXF wind-stress binaries: ustress.bin / vstress.bin.

    Expects a GRIDDED_2D_SURFACE bundle carrying `tau_x` / `tau_y` (Pa) —
    produced by the `wind_to_stress` Layer C operator (Large & Pond 1981).
    EXF runs in direct-stress mode (readStressOnAgrid), so the bundle must
    already hold pre-computed stress, not raw wind.

    Canonical var → MIT file map:
      tau_x → ustress.bin
      tau_y → vstress.bin

    NaN land cells are filled with 0 (no momentum forcing over land).
    """

    capabilities = ConsumerCapabilities(
        accepts_location_kinds=(LocationKind.GRIDDED_2D_SURFACE,),
        required_convention=ConventionTag.MITGCM,
        required_variables=(),
        artifact_kind=ArtifactKind.FILE_BINARY,
        artifact_schema_version="0.1.0",
    )

    _VAR_TO_FILE = {
        "tau_x": "ustress.bin",
        "tau_y": "vstress.bin",
    }

    def __init__(self, ctx: ConsumerContext, *, dtype: str = "float64", **_: Any) -> None:
        self._ctx = ctx
        self._dtype = dtype

    def consume(self, bundle: DataBundle) -> ArtifactRef:
        self._pre_validate(bundle)
        out_files: list[tuple[str, str]] = []
        input_dir = self._ctx.output_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        for var in bundle.meta.variables:
            fname = self._VAR_TO_FILE.get(var.name)
            if fname is None:
                # Unknown var (e.g. legacy raw wind) — skip; only stress is exported.
                continue
            arr = np.asarray(bundle.data[var.name].values).astype(np.float64)
            # Land NaN → 0 (no wind stress over land)
            if np.isnan(arr).any():
                arr = np.nan_to_num(arr, nan=0.0)
            fpath = input_dir / fname
            write_bin_atomic(arr, fpath, dtype=self._dtype, endianness="big")
            dim_list_meta = [(s, 1, s) for s in arr.shape]
            write_meta_companion(
                fpath.with_suffix(".meta"),
                dim_list=dim_list_meta,
                dataprec=self._dtype,
                nrecords=arr.shape[0] if arr.ndim > 0 else 1,
            )
            out_files.append((var.name, fname))

        if not out_files:
            raise ValueError(
                "mitgcm_exf_binary: no tau_x/tau_y in bundle — run wind_to_stress "
                f"operator first (have {list(bundle.data.data_vars)})"
            )

        # Index: one ``<var>\t<file>`` line per binary written into input_dir.
        primary = input_dir / "exf_index.txt"
        primary.write_text("\n".join(f"{v}\t{f}" for v, f in out_files) + "\n")
        return ArtifactRef(
            name="mitgcm_exf_binary",
            path=primary,
            artifact_kind=ArtifactKind.FILE_BINARY,
            bundle_hash=bundle.meta.provenance[-1].bundle_hash if bundle.meta.provenance else "00000000",
            schema_version="0.1.0",
            extras={"n_files": len(out_files), "files": [v for (v, _) in out_files]},
        )

    def _pre_validate(self, bundle: DataBundle) -> None:
        if bundle.meta.location_kind != LocationKind.GRIDDED_2D_SURFACE:
            raise ValueError(f"mitgcm_exf_binary: requires GRIDDED_2D_SURFACE; got {bundle.meta.location_kind}")
        if bundle.meta.convention != ConventionTag.MITGCM:
            raise ValueError(f"mitgcm_exf_binary: requires MITGCM; got {bundle.meta.convention}")


# ============================================================
# mitgcm_size_h (Profile M)
# ============================================================


@register_consumer("mitgcm_size_h")
class MitgcmSizeHAdapter:
    """Generates code/SIZE.h by wrapping legacy ``_write_size_h``."""

    capabilities = ConsumerCapabilities(
        accepts_location_kinds=(LocationKind.MESH,),
        required_convention=ConventionTag.OPAQUE,  # SIZE.h is text generation; convention irrelevant
        required_variables=(),
        artifact_kind=ArtifactKind.FILE_TEXT,
        artifact_schema_version="0.1.0",
    )

    def __init__(self, ctx: ConsumerContext, *, ol_x: int = 4, ol_y: int = 4, **_: Any) -> None:
        # Nr is derived from ModelInput.nz via _assemble_legacy_export_args (see
        # consume: args.nz); no `nz` constructor param is read.
        self._ctx = ctx
        self._ol_x = ol_x
        self._ol_y = ol_y

    def consume(self, bundle: DataBundle) -> ArtifactRef:
        code_dir = self._ctx.output_dir / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        size_h_path = code_dir / "SIZE.h"
        args = _assemble_legacy_export_args(self._ctx)
        # compute_mpi_topology operator stamps the decomposition into
        # bundle.meta.extras['mpi_topology']; consume it when present so the
        # resulting SIZE.h matches mit_parallel.{nPx,nPy} from yaml.  Without
        # this the generated header was always single-process and stage3
        # mpirun -np N would either fail or fork N independent serial copies.
        topo = bundle.meta.extras.get("mpi_topology") or {}
        n_px = int(topo.get("nPx", 1))
        n_py = int(topo.get("nPy", 1))
        sNx = topo.get("sNx")
        sNy = topo.get("sNy")
        ol_x = int(topo.get("halo", self._ol_x))
        ol_y = int(topo.get("halo", self._ol_y))
        _legacy_export_module()._write_size_h(
            size_h_path,
            nx=args.nx,
            ny=args.ny,
            nz=args.nz,
            n_px=n_px,
            n_py=n_py,
            sNx=sNx,
            sNy=sNy,
            ol_x=ol_x,
            ol_y=ol_y,
        )

        return ArtifactRef(
            name="mitgcm_size_h",
            path=size_h_path,
            artifact_kind=ArtifactKind.FILE_TEXT,
            bundle_hash=_bundle_hash(bundle),
            schema_version="0.1.0",
            extras={
                "nx": args.nx, "ny": args.ny, "nz": args.nz,
                "nPx": n_px, "nPy": n_py,
                "wrapped": "_write_size_h",
            },
        )


# ============================================================
# mitgcm_topology (Profile M)
# ============================================================


@register_consumer("mitgcm_topology")
class MitgcmTopologyAdapter:
    """Writes mit_topology.yaml documenting the MPI decomposition decision."""

    capabilities = ConsumerCapabilities(
        accepts_location_kinds=(LocationKind.MESH,),
        required_convention=ConventionTag.OPAQUE,
        required_variables=(),
        artifact_kind=ArtifactKind.FILE_TEXT,
        artifact_schema_version="0.1.0",
    )

    def __init__(self, ctx: ConsumerContext, **_: Any) -> None:
        self._ctx = ctx

    def consume(self, bundle: DataBundle) -> ArtifactRef:
        topo = bundle.meta.extras.get("mpi_topology")
        if topo is None:
            raise ValueError("mitgcm_topology: bundle missing 'mpi_topology' extras")
        meta_dir = self._ctx.output_dir / "_pipeline_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        path = meta_dir / "mit_topology.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(dict(topo), sort_keys=False))
        return ArtifactRef(
            name="mitgcm_topology",
            path=path,
            artifact_kind=ArtifactKind.FILE_TEXT,
            bundle_hash=bundle.meta.provenance[-1].bundle_hash if bundle.meta.provenance else "00000000",
            schema_version="0.1.0",
            extras=dict(topo),
        )


# ============================================================
# mitgcm_data_namelist (Profile M)
# ============================================================


@register_consumer("mitgcm_data_namelist")
class MitgcmDataNamelistAdapter:
    """Generates input/data by wrapping legacy ``_write_data``."""

    capabilities = ConsumerCapabilities(
        accepts_location_kinds=tuple(LocationKind),
        required_convention=ConventionTag.OPAQUE,
        required_variables=(),
        artifact_kind=ArtifactKind.FILE_TEXT,
        artifact_schema_version="0.1.0",
    )

    def __init__(self, ctx: ConsumerContext, **_: Any) -> None:
        self._ctx = ctx

    def consume(self, bundle: DataBundle) -> ArtifactRef:
        out_dir = self._ctx.output_dir / "input"
        out_dir.mkdir(parents=True, exist_ok=True)

        data_path = out_dir / "data"
        args = _assemble_legacy_export_args(self._ctx)
        run_cfg = self._ctx.run_cfg
        legacy = _legacy_export_module()
        # Bottom friction: the three terms of mom_u_botdrag_coeff.F come from the
        # LOCOMgcm bottom_drag dynamics module (layer_d/_mit_switches.py), not
        # from template constants.

        no_slip_bottom = switch_value(run_cfg, _SWITCH_BY_NAME["no_slip_bottom"])
        bottom_drag_linear, bottom_drag_quadratic = bottom_drag_coefficients(run_cfg)
        if no_slip_bottom:
            module_viscosity = float(
                (bottom_drag_module_kwargs(run_cfg) or {}).get("vertical_viscosity", 0.0) or 0.0
            )
            if module_viscosity != float(args.vertical_viscosity):
                raise ValueError(
                    "no_slip_bottom is active but LOCOMgcm's bottom_drag."
                    f"vertical_viscosity={module_viscosity!r} differs from the "
                    f"viscAz the export writes ({args.vertical_viscosity!r}, the "
                    "stepper's implicit_viscosity). MITgcm builds the no-slip "
                    "friction from kappaRU=viscAr (mom_u_botdrag_coeff.F:127), so "
                    "the namelist cannot express two values; make them equal."
                )
        legacy._write_data(
            data_path,
            nx=args.nx,
            ny=args.ny,
            nz=args.nz,
            resolution_deg=args.resolution_deg,
            depth_m=args.depth_m,
            layer_thickness=args.layer_thickness,
            dt_seconds=args.dt_seconds,
            dt_mom_seconds=args.dt_mom_seconds,
            dt_tracer_seconds=args.dt_tracer_seconds,
            dt_freesurf_seconds=args.dt_freesurf_seconds,
            dt_clock_seconds=args.dt_clock_seconds,
            total_seconds=args.total_seconds,
            viscosity=args.viscosity,
            viscA4=args.viscA4,
            diffusivity_t=args.diffusivity_t,
            diffusivity_s=args.diffusivity_s,
            diffusivity_k4_temp=tracer_biharmonic_diffusivity(run_cfg, "temp"),
            diffusivity_k4_salt=tracer_biharmonic_diffusivity(run_cfg, "salt"),
            xg_origin=args.xg_origin,
            yg_origin=args.yg_origin,
            dump_freq=args.dump_freq_seconds,
            init_temp=args.init_temp,
            init_salt=args.init_salt,
            tempAdvScheme=args.tempAdvScheme,
            saltAdvScheme=args.saltAdvScheme,
            multi_dim_advection=args.multi_dim_advection,
            tempVertAdvScheme=args.tempVertAdvScheme,
            saltVertAdvScheme=args.saltVertAdvScheme,
            temperature_implicit_vertical_advection=(
                args.temperature_implicit_vertical_advection
            ),
            salinity_implicit_vertical_advection=(
                args.salinity_implicit_vertical_advection
            ),
            momentum_implicit_vertical_advection=(
                args.momentum_implicit_vertical_advection
            ),
            eos=args.eos,
            rho_const=rho_const_value(run_cfg),
            gravity=gravity_value(run_cfg),
            use_cartesian=args.use_cartesian,
            implicit_diffusion=args.implicit_diffusion,
            tracer_vertical_diffusivity_kr=args.tracer_vertical_diffusivity_kr,
            implicit_viscosity=args.implicit_viscosity,
            no_slip_sides=args.no_slip_sides,
            no_slip_bottom=no_slip_bottom,
            bottom_drag_linear=bottom_drag_linear,
            bottom_drag_quadratic=bottom_drag_quadratic,
            vertical_viscosity=args.vertical_viscosity,
            hfac_min=args.hfac_min,
            hfac_min_dz=args.hfac_min_dz,
            lin_fs_conserve_tr=args.lin_fs_conserve_tr,
            ke_scheme=args.ke_scheme,
            viscosity_scheme=args.viscosity_scheme,
            visc_c2_leith=args.visc_c2_leith,
            visc_c2_smag=args.visc_c2_smag,
            momentum_advection_form=args.momentum_advection_form,
            nonhydrostatic=args.nonhydrostatic,
            cg3d_max_iters=args.cg3d_max_iters,
            cg3d_target_residual=args.cg3d_target_residual,
            cg3d_target_res_w_unit=args.cg3d_target_res_w_unit,
            nh_am2=args.nh_am2,
            # PARM03 AB forcing placement: single-sourced from run_cfg via
            # layer_d/_mit_switches.py.  A hardcoded `forcing_In_AB=.FALSE.`
            # would pin both integers, so a case declaring
            # mom_forcing_out_ab: false would run MITgcm in the opposite regime
            # without a signal.
            mom_forcing_out_ab=switch_value(run_cfg, _SWITCH_BY_NAME["momForcingOutAB"]),
            trac_forcing_out_ab=switch_value(run_cfg, _SWITCH_BY_NAME["tracForcingOutAB"]),
            # Time-stepping switches: the stepper's exact_conserv and
            # stagger_tracer_stepping, and the tracer equations LOCOMgcm steps
            # (the tracer modules' tracer_names).
            exact_conserv=switch_value(run_cfg, _SWITCH_BY_NAME["exactConserv"]),
            stagger_time_step=switch_value(run_cfg, _SWITCH_BY_NAME["staggerTimeStep"]),
            temp_stepping=switch_value(run_cfg, _SWITCH_BY_NAME["tempStepping"]),
            salt_stepping=switch_value(run_cfg, _SWITCH_BY_NAME["saltStepping"]),
            # Grid viscosity limits, Leith-D / full Leith, biharmonic Leith/Smag
            # and sideDragFactor from LOCOMgcm's horizontal_viscosity module.
            viscosity_extra_lines=render_viscosity_lines(run_cfg),
            # The stepper's ab_eps (its default when the yaml omits it), not a
            # template constant: the Gong mitcheck case runs LOCOMgcm at 0.02.
            ab_eps=ab_eps_value(run_cfg),
            implic_surf_press=float(stepper_field(run_cfg, "implicit_surf_press")),
            implic_div_2d_flow=float(stepper_field(run_cfg, "implicit_div_2d_flow")),
            mom_stepping=bool(stepper_field(run_cfg, "mom_stepping")),
            cg2d=cg2d_namelist_values(run_cfg),
            c_adj_freq=convective_adjustment_frequency(run_cfg),
        )
        # Re-parse what was actually written, rather than trusting the kwargs
        # above: this is the check the OBCSfixTopo divergence lacked.
        written = data_path.read_text()
        assert_switches_agree(run_cfg, "data", written)
        assert_viscosity_agrees(run_cfg, written)
        assert_ab_eps_agrees(run_cfg, written)
        # Every PARM01-PARM04 key against the model (DATA_KEYS).
        assert_data_keys_agree(
            DataView(run_cfg=run_cfg, params=_read_model_input_run_params(self._ctx),
                     args=args, eos=args.eos),
            written,
        )

        return ArtifactRef(
            name="mitgcm_data_namelist",
            path=data_path,
            artifact_kind=ArtifactKind.FILE_TEXT,
            bundle_hash=_bundle_hash(bundle),
            schema_version="0.1.0",
            extras={
                "input_dir": str(out_dir),
                "files_written": ["data"],
                "wrapped": "_write_data",
            },
        )


# ============================================================
# data.pkg / packages.conf / data.obcs
# ============================================================


@register_consumer("mitgcm_pkg_namelist")
class MitgcmPkgNamelistAdapter:
    """Writes input/data.pkg by wrapping legacy ``_write_data_pkg``."""

    capabilities = ConsumerCapabilities(
        accepts_location_kinds=tuple(LocationKind),
        required_convention=ConventionTag.OPAQUE,
        required_variables=(),
        artifact_kind=ArtifactKind.FILE_TEXT,
        artifact_schema_version="0.1.0",
    )

    def __init__(self, ctx: ConsumerContext, *,
                 use_obcs: bool = True,  # accepted for backward-compat, currently ignored (consume only forwards use_exf)
                 use_exf: bool = True,
                 use_cal: bool = True,
                 **_: Any) -> None:
        self._ctx = ctx
        self._use_obcs = use_obcs
        self._use_exf = use_exf
        self._use_cal = use_cal

    def consume(self, bundle: DataBundle) -> ArtifactRef:
        out_dir = self._ctx.output_dir / "input"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "data.pkg"
        use_kpp = switch_value(self._ctx.run_cfg, _SWITCH_BY_NAME["useKPP"])
        use_diagnostics = _section_transport_enabled(self._ctx.run_cfg)
        # Honor the consumer's use_exf flag (previously ignored — data.pkg always
        # emitted useEXF=.TRUE.). Default True retains that setting for wind-forced
        # cases; no-wind cases (Gong) set use_exf:false in yaml.
        _legacy_export_module()._write_data_pkg(
            path,
            use_exf=self._use_exf,
            use_kpp=use_kpp,
            use_diagnostics=use_diagnostics,
        )
        assert_switches_agree(self._ctx.run_cfg, "data.pkg", path.read_text())
        return ArtifactRef(
            name="mitgcm_pkg_namelist", path=path,
            artifact_kind=ArtifactKind.FILE_TEXT,
            bundle_hash=_bundle_hash(bundle), schema_version="0.1.0",
            extras={"input_dir": str(out_dir), "wrapped": "_write_data_pkg",
                    "use_exf": self._use_exf, "use_kpp": use_kpp,
                    "use_diagnostics": use_diagnostics},
        )


@register_consumer("mitgcm_kpp_namelist")
class MitgcmKppNamelistAdapter:
    """Writes input/data.kpp using MITgcm's stock KPP options and defaults."""

    capabilities = ConsumerCapabilities(
        accepts_location_kinds=tuple(LocationKind),
        required_convention=ConventionTag.OPAQUE,
        required_variables=(),
        artifact_kind=ArtifactKind.FILE_TEXT,
        artifact_schema_version="0.1.0",
    )

    def __init__(self, ctx: ConsumerContext, **_: Any) -> None:
        self._ctx = ctx

    def consume(self, bundle: DataBundle) -> ArtifactRef:
        out_dir = self._ctx.output_dir / "input"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "data.kpp"
        _legacy_export_module()._write_data_kpp(path)
        return ArtifactRef(
            name="mitgcm_kpp_namelist",
            path=path,
            artifact_kind=ArtifactKind.FILE_TEXT,
            bundle_hash=_bundle_hash(bundle),
            schema_version="0.1.0",
            extras={"input_dir": str(out_dir), "wrapped": "_write_data_kpp"},
        )


@register_consumer("mitgcm_diagnostics_namelist")
class MitgcmDiagnosticsNamelistAdapter:
    """Writes input/data.diagnostics for configured section observations."""

    capabilities = ConsumerCapabilities(
        accepts_location_kinds=tuple(LocationKind),
        required_convention=ConventionTag.OPAQUE,
        required_variables=(),
        artifact_kind=ArtifactKind.FILE_TEXT,
        artifact_schema_version="0.1.0",
    )

    def __init__(self, ctx: ConsumerContext, **_: Any) -> None:
        self._ctx = ctx

    def consume(self, bundle: DataBundle) -> ArtifactRef:
        if not _section_transport_enabled(self._ctx.run_cfg):
            raise ValueError(
                "mitgcm_diagnostics_namelist requires observations.section_transport_u"
            )
        out_dir = self._ctx.output_dir / "input"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "data.diagnostics"
        _legacy_export_module()._write_data_diagnostics(path)
        return ArtifactRef(
            name="mitgcm_diagnostics_namelist",
            path=path,
            artifact_kind=ArtifactKind.FILE_TEXT,
            bundle_hash=_bundle_hash(bundle),
            schema_version="0.1.0",
            extras={"input_dir": str(out_dir), "wrapped": "_write_data_diagnostics"},
        )


@register_consumer("mitgcm_packages_conf")
class MitgcmPackagesConfAdapter:
    """Writes code/packages.conf by wrapping legacy ``_write_packages_conf``."""

    capabilities = ConsumerCapabilities(
        accepts_location_kinds=tuple(LocationKind),
        required_convention=ConventionTag.OPAQUE,
        required_variables=(),
        artifact_kind=ArtifactKind.FILE_TEXT,
        artifact_schema_version="0.1.0",
    )

    def __init__(self, ctx: ConsumerContext, *,
                 packages: tuple[str, ...] = ("gfd", "obcs", "exf", "cal"),
                 **_: Any) -> None:
        self._ctx = ctx
        self._packages = tuple(packages)

    def consume(self, bundle: DataBundle) -> ArtifactRef:
        code_dir = self._ctx.output_dir / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        path = code_dir / "packages.conf"
        # Compile exf only when the consumer's packages list includes it
        # (previously ignored — packages.conf always compiled exf). Every
        # existing case lists exf so the build is unchanged; Gong omits it.
        include_exf = "exf" in self._packages
        # kpp is *derived*, not merely listed: data.pkg's useKPP comes from
        # model.stepper.kwargs.kpp_enabled (_mit_switches.py:879-892), and a
        # useKPP=.TRUE. namelist against a build without ALLOW_KPP ends the run
        # at packages_check.F:245-248 -> packages_error_msg.F:61-62.  The two
        # files must therefore not be able to disagree.  The union keeps the
        # legal opposite direction (compiled but not activated, a weak unused-
        # package message at kpp_readparms.F:59-66) expressible through the
        # packages tuple.
        include_kpp = "kpp" in self._packages or switch_value(
            self._ctx.run_cfg, _SWITCH_BY_NAME["useKPP"]
        )
        diagnostics = _section_transport_enabled(self._ctx.run_cfg)
        _legacy_export_module()._write_packages_conf(
            path,
            include_exf=include_exf,
            include_kpp=include_kpp,
            diagnostics=diagnostics,
        )
        return ArtifactRef(
            name="mitgcm_packages_conf", path=path,
            artifact_kind=ArtifactKind.FILE_TEXT,
            bundle_hash=_bundle_hash(bundle), schema_version="0.1.0",
            extras={"code_dir": str(code_dir), "wrapped": "_write_packages_conf",
                    "include_exf": include_exf, "include_kpp": include_kpp,
                    "diagnostics": diagnostics},
        )


def _assert_obcs_sponge_consistency(
    run_cfg,
    *,
    u_bound: float,
    u_inner: float,
    v_bound: float,
    v_inner: float,
    sponge_width: int,
) -> None:
    """Fail loudly if the MIT data.obcs sponge relax timescales just written
    disagree with the LOCOMgcm model.dynamics sponge config.

    Root cause this gate prevents: the OBCS sponge
    relax timescale is authored twice — once in ``boundary.sponge_*`` (-> MIT
    data.obcs Urelax/Vrelax) and once in ``model.dynamics.*_sponge_relaxation``
    (-> LOCOMgcm runtime) — with nothing linking them, so they silently drifted (LOCOMgcm
    constant 44714 vs MIT gradient 22357->44714). Identical *inputs* then relax
    differently inside the 50-cell band -> O(0.25 m/s) in-band divergence that
    op-level tests miss. MIT obcs_sponge.F is orientation-based (E/W use
    Urelax{bound,inner}, N/S use Vrelax{bound,inner}, applied to u/v/T/S; see
    MITgcm pkg/obcs/obcs_sponge.F); LOCOMgcm encodes the same via
    ``timescale_seconds*`` (E/W) + ``timescale_seconds_ns*`` (N/S). Bypass with
    DEFGCM_SKIP_SPONGE_GATE=1 (only for intentional LOCOMgcm!=MIT experiments).
    """
    import os

    if os.environ.get("DEFGCM_SKIP_SPONGE_GATE") == "1":
        return
    dyn = getattr(getattr(run_cfg, "model", None), "dynamics", ()) or ()
    targets = {
        "momentum_boundary_sponge_relaxation",
        "tracer_boundary_sponge_relaxation",
    }
    rtol = 1e-4
    problems: list[str] = []
    for entry in dyn:
        if entry.name not in targets:
            continue
        kw = dict(entry.kwargs)
        if str(kw.get("sponge_mode", "mitgcm")) != "mitgcm":
            continue  # linear mode uses a different single-tau profile
        ts_b = float(kw["timescale_seconds"])
        ts_i = kw.get("timescale_inner_seconds")
        ts_i = 10.0 * ts_b if ts_i is None else float(ts_i)
        ns_b = kw.get("timescale_seconds_ns")
        ns_b = ts_b if ns_b is None else float(ns_b)
        ns_i = kw.get("timescale_inner_seconds_ns")
        ns_i = ts_i if ns_i is None else float(ns_i)
        w = int(kw.get("width", sponge_width))
        for label, to_val, mit_val in (
            ("E/W (Urelax) bound", ts_b, u_bound),
            ("E/W (Urelax) inner", ts_i, u_inner),
            ("N/S (Vrelax) bound", ns_b, v_bound),
            ("N/S (Vrelax) inner", ns_i, v_inner),
            ("sponge width", float(w), float(sponge_width)),
        ):
            if abs(to_val - mit_val) / max(abs(mit_val), 1e-12) > rtol:
                problems.append(
                    f"{entry.name}: {label} TO={to_val:g} != MIT data.obcs={mit_val:g}"
                )
    if problems:
        raise ValueError(
            "OBCS sponge config drift between LOCOMgcm and the exported MIT "
            "data.obcs: the two models would relax their sponge bands "
            "differently (silent in-band divergence). Make the model.dynamics "
            "sponge timescales match boundary.sponge_* / the per-U/V seconds "
            "fields. Mismatches:\n  " + "\n  ".join(problems)
            + "\n(set DEFGCM_SKIP_SPONGE_GATE=1 to bypass for intentional "
            "TO!=MIT sponge experiments)"
        )


# ---------------------------------------------------------------------------
# data.obcs OB_J*/OB_I* index tables
#
# MITgcm's open boundary is a per-column / per-row index table, not an edge:
# ``OB_Jsouth(i)`` carries ``Nx`` entries, ``OB_Ieast(j)`` carries ``Ny``, and an
# entry of ``0`` is ``OB_indexUnset`` (``pkg/obcs/obcs_readparms.F:238``), which
# switches every OBCS action off on that column.  The runtime momentum sponge
# implements the same guard.  Both ends resolve the table from ONE case field,
# ``boundary.ob_index_mode`` — see ``defgcm/boundaries/open.py``
# ``resolve_ob_along_masks`` — and :func:`_assert_ob_index_agreement` re-parses
# the emitted namelist to prove the file on disk still says what the resolver
# said, the same way ``assert_switches_agree`` does for the logicals.
# ---------------------------------------------------------------------------

#: ``(yaml side letter, namelist keyword, BoundarySide value)``, in the emission
#: order the legacy exporter used.
_OB_INDEX_SPECS: tuple[tuple[str, str, str], ...] = (
    ("S", "OB_Jsouth", "south"),
    ("N", "OB_Jnorth", "north"),
    ("W", "OB_Iwest", "west"),
    ("E", "OB_Ieast", "east"),
)

#: Ignition counter for :func:`_assert_ob_index_agreement`.  A test that only
#: inspects the emitted text cannot tell whether the agreement gate ran at all;
#: this records whether the declared path actually executed.
OB_INDEX_AGREEMENT_COUNTS: dict[str, int] = {"checks": 0, "sides": 0}

#: Namelist value column: ``" OB_Jsouth = "`` and ``" OB_Iwest  = "`` — the
#: two-space padding of the shorter keywords is the legacy exporter's, kept so a
#: ``full_edge`` case still writes a byte-for-byte identical data.obcs.
_OB_KEYWORD_WIDTH = 9

#: Wrap the value list at this column.  A namelist value list may span records
#: freely, so a large wet-only table can occupy several lines.
_OB_LINE_WIDTH = 72


def _ob_side_facts(side: str, *, nx: int, ny: int) -> tuple[int, int]:
    """``(table length, index value of an OPEN entry)`` for one side.

    The open value is the absolute boundary index MITgcm resolves to: south/west
    are row/column 1, north is ``Ny`` and east is ``Nx``.  MITgcm also accepts the
    relative spelling (``-1``, converted at ``obcs_readparms.F:672-676``); the
    absolute form is what this exporter has always written.
    """
    if side in ("south", "north"):
        return nx, (1 if side == "south" else ny)
    return ny, (1 if side == "west" else nx)


def _read_model_input_center_mask(ctx: ConsumerContext, *, nx: int, ny: int) -> Any:
    """The runtime grid's own ``masks.center``, as written by ``model_input``.

    🔴 Single source.  ``io/runner/model_input.py`` stores ``grid.masks.center``
    into ``_pipeline_meta/model_input/grid/tensors.npz`` as ``mask_center``.  This
    is the **equivalent mask produced by the same factory from the same config**
    as the one the LOCOMgcm run gates on — precisely, stage1a and stage2 each
    call ``build_grid_from_run_cfg`` on the same ``run_cfg``, so the two are two
    constructions of one specification rather than literally one array, and
    nothing currently cross-checks them (a pre-existing risk surface of the same
    shape as the ``OBCSfixTopo`` divergence, not introduced here).

    Re-thresholding bathymetry here instead would be strictly worse: a second,
    silently *different* definition of "wet", since ``hFacMin`` clamping can leave
    ``Depth > 0`` on a column whose every cell is dry.
    """
    import torch

    _scalars, tensors = _read_model_input_grid(ctx)
    if "mask_center" not in tensors:
        raise ValueError(
            "mitgcm_obcs_namelist: boundary.ob_index_mode='wet_only' needs the "
            "runtime grid's center wet mask, but "
            "_pipeline_meta/model_input/grid/tensors.npz has no 'mask_center'"
        )
    mask = torch.as_tensor(np.asarray(tensors["mask_center"]))
    if tuple(mask.shape[-2:]) != (ny, nx):
        raise ValueError(
            f"mitgcm_obcs_namelist: model_input mask_center has shape "
            f"{tuple(mask.shape)}, incompatible with grid ({ny}, {nx})"
        )
    return mask


def _ob_index_values(
    ctx: ConsumerContext, run_cfg: Any, *, nx: int, ny: int, side_set: set[str]
) -> tuple[Any, dict[str, tuple[str, list[int]]]]:
    """Resolve ``(tables, {side letter: (keyword, per-column index values)})``.

    The dict holds only the sides that will actually be emitted; ``tables`` covers
    all four, so :func:`_assert_ob_index_agreement` can also check the sides that
    are *not* emitted (see its docstring — that is where a runtime/export side-set
    disagreement would otherwise hide).

    ``full_edge`` never reads geometry: the resolver returns an ungated table for
    every side and each entry is the side's open index, which run-length encodes
    back to the historical ``{Nx}*1`` form.
    """
    from defgcm.boundaries.open import (
        BoundarySide,
        case_ob_policy,
        resolve_ob_along_masks_from_center_mask,
    )

    # The policy the sponge modules apply (their own ob_index_mode / ob_sides, else
    # the boundary fields), not the boundary fields alone: a module override would
    # otherwise gate a column the exported table opens or closes differently.
    mode, gated = case_ob_policy(run_cfg.model.dynamics, run_cfg.boundary)
    emitted = tuple(spec for spec in _OB_INDEX_SPECS if spec[0] in side_set)
    stray = sorted(side.value for side in set(gated) - {BoundarySide(spec[2]) for spec in emitted})
    if stray:
        raise ValueError(
            f"mitgcm_obcs_namelist: the sponges gate on OB tables of {stray}, which "
            "boundary.sides gives no open boundary in data.obcs"
        )
    center = (
        None
        if mode == "full_edge"
        else _read_model_input_center_mask(ctx, nx=nx, ny=ny)
    )
    # The resolver gates the emitted sides the sponges gate (their ob_sides); an
    # emitted side they leave ungated keeps the open table.
    tables = resolve_ob_along_masks_from_center_mask(
        mode, center, tuple(spec[2] for spec in emitted if BoundarySide(spec[2]) in gated)
    )

    resolved: dict[str, tuple[str, list[int]]] = {}
    for letter, keyword, side in emitted:
        length, open_value = _ob_side_facts(side, nx=nx, ny=ny)
        table = tables.get(BoundarySide(side))
        if table is None:
            values = [open_value] * length
        else:
            if int(table.shape[0]) != length:
                raise ValueError(
                    f"mitgcm_obcs_namelist: {keyword} table has "
                    f"{int(table.shape[0])} entries, expected {length}"
                )
            values = [open_value if bool(v) else 0 for v in table.tolist()]
        resolved[letter] = (keyword, values)
    return tables, resolved


def _render_ob_index_line(keyword: str, values: list[int]) -> str:
    """Render one ``OB_*`` table as run-length compressed namelist text."""
    runs: list[list[int]] = []
    for value in values:
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])
    # Always the ``count*value`` form, including for a run of one: it is valid
    # namelist syntax and it keeps the single-run case byte-identical to the
    # historical ``f" OB_Jsouth = {nx}*1,"``.
    tokens = [f"{count}*{value}" for value, count in runs]

    prefix = f" {keyword:<{_OB_KEYWORD_WIDTH}} = "
    indent = " " * len(prefix)
    lines: list[str] = []
    current = prefix
    for i, token in enumerate(tokens):
        piece = token + ("," if i == len(tokens) - 1 else ", ")
        if current.strip() and len(current) + len(piece) > _OB_LINE_WIDTH:
            lines.append(current.rstrip())
            current = indent
        current += piece
    lines.append(current.rstrip())
    return "\n".join(lines)


def _parse_ob_index_values(text: str, keyword: str) -> list[int] | None:
    """Re-parse an emitted ``OB_*`` table back into per-column integers.

    Returns ``None`` when the keyword is absent.  Accepts the multi-record value
    lists Fortran namelists allow, so the wrapped form round-trips.
    """
    import re

    lines = text.splitlines()
    head = re.compile(rf"^\s*{re.escape(keyword)}\s*=\s*(.*)$")
    boundary = re.compile(r"^\s*(?:[&/#!]|[A-Za-z_]\w*\s*(?:\(\d+\))?\s*=)")
    for index, line in enumerate(lines):
        match = head.match(line)
        if match is None:
            continue
        chunks = [match.group(1)]
        for cont in lines[index + 1:]:
            if not cont.strip() or boundary.match(cont):
                break
            chunks.append(cont)
        values: list[int] = []
        for token in " ".join(chunks).replace(",", " ").split():
            if "*" in token:
                count, _, value = token.partition("*")
                values.extend([int(value)] * int(count))
            else:
                values.append(int(token))
        return values
    return None


def _assert_ob_index_agreement(
    text: str, resolved: dict[str, tuple[str, list[int]]], tables: Any
) -> None:
    """Assembly-time gate: the emitted OB tables must equal the resolved tables.

    Re-parses the namelist rather than trusting the renderer, so a hand-edited
    f-string or a broken run-length encoder is an export-time exception instead
    of a reference model that opens different boundary columns than the
    LOCOMgcm run it is compared against.

    🔴 All **four** sides are checked, not just the emitted ones.  A gate that
    only walked ``resolved`` would be structurally blind to the very divergence it
    exists to catch: a side that ``boundary.sides`` omits produces no namelist
    line, so if the resolver nevertheless returned a table for it the two ends
    would disagree about that side in silence.  For a non-emitted side the
    assertion is therefore the pair "no ``OB_*`` line in the file" **and**
    "resolver table is ``None``" — which is also what the runtime sponge sees for
    that side, since it is handed the same ``boundary.sides``.
    """
    from defgcm.boundaries.open import BoundarySide

    problems: list[str] = []
    for letter, keyword, side in _OB_INDEX_SPECS:
        if letter in resolved:
            continue
        parsed = _parse_ob_index_values(text, keyword)
        if parsed is not None:
            problems.append(
                f"{keyword}: side {side!r} is not in boundary.sides, yet "
                f"data.obcs carries {len(parsed)} entries for it"
            )
        if tables.get(BoundarySide(side)) is not None:
            problems.append(
                f"{keyword}: side {side!r} is not in boundary.sides, yet the "
                "resolver produced a table for it — the runtime sponge would gate "
                "a side the reference model has no open boundary on"
            )
    for keyword, expected in resolved.values():
        parsed = _parse_ob_index_values(text, keyword)
        if parsed is None:
            problems.append(
                f"{keyword}: resolved to {len(expected)} entries but the keyword "
                "is absent from the rendered data.obcs"
            )
            continue
        if len(parsed) != len(expected):
            problems.append(
                f"{keyword}: data.obcs expands to {len(parsed)} entries, the "
                f"resolver produced {len(expected)}"
            )
            continue
        differing = [i for i, (p, e) in enumerate(zip(parsed, expected)) if p != e]
        if differing:
            open_flips = sum(
                1 for i in differing if (parsed[i] != 0) != (expected[i] != 0)
            )
            head = ", ".join(
                f"[{i}] namelist={parsed[i]} resolver={expected[i]}"
                for i in differing[:5]
            )
            problems.append(
                f"{keyword}: {len(differing)} of {len(expected)} entries differ "
                f"({open_flips} of them flip the column open/closed); {head}"
            )
    if problems:
        raise ValueError(
            "MIT/TO open-boundary index-table divergence while writing data.obcs: "
            "the exported reference model would open a different set of boundary "
            "columns than the LOCOMgcm run it is compared against (the tables are "
            "resolved from the policy the runtime sponges apply, case_ob_policy).\n  "
            + "\n  ".join(problems)
        )
    OB_INDEX_AGREEMENT_COUNTS["checks"] += 1
    # Always four: the count is of sides *checked*, and a side being absent from
    # boundary.sides is itself one of the checked properties.
    OB_INDEX_AGREEMENT_COUNTS["sides"] += len(_OB_INDEX_SPECS)


@register_consumer("mitgcm_obcs_namelist")
class MitgcmObcsNamelistAdapter:
    """Writes input/data.obcs — open boundary configuration namelist.

    Reads sides from run_cfg.boundary.sides; uses sponge from
    run_cfg.boundary.sponge_*. OB_*file names map to mitgcm_obc_binary
    consumer outputs.
    """

    capabilities = ConsumerCapabilities(
        accepts_location_kinds=tuple(LocationKind),
        required_convention=ConventionTag.OPAQUE,
        required_variables=(),
        artifact_kind=ArtifactKind.FILE_TEXT,
        artifact_schema_version="0.1.0",
    )

    def __init__(self, ctx: ConsumerContext, **_: Any) -> None:
        self._ctx = ctx

    def consume(self, bundle: DataBundle) -> ArtifactRef:
        run_cfg = self._ctx.run_cfg
        if run_cfg is None:
            raise ValueError("mitgcm_obcs_namelist: ctx.run_cfg is None")

        out_dir = self._ctx.output_dir / "input"
        out_dir.mkdir(parents=True, exist_ok=True)
        nx = run_cfg.grid.nx
        ny = run_cfg.grid.ny

        # OBCS_monitorFreq = obcs period (legacy mitgcm_export._write_data_obcs).
        obcs_period, _repeat, _n = _extract_obcs_timing(bundle)

        # Sponge timescales (seconds). inner = inner-edge timescale, bound =
        # boundary timescale. MITgcm obcs_sponge.F is orientation-dependent
        # (E/W use Urelaxobcs{bound,inner}, N/S use Vrelaxobcs{bound,inner}; see
        # MITgcm pkg/obcs/obcs_sponge.F). Prefer the per-U/V
        # seconds fields when set (so reference values like Gong's 44.714 s pass
        # through exactly); else fall back to the single hours-based pair (U==V).
        _bcfg = run_cfg.boundary
        ts_bound = _bcfg.sponge_timescale_hours * 3600.0
        ts_inner = _bcfg.sponge_timescale_inner_hours * 3600.0
        u_relax_bound = _bcfg.sponge_u_timescale_bound_seconds
        u_relax_bound = ts_bound if u_relax_bound is None else u_relax_bound
        u_relax_inner = _bcfg.sponge_u_timescale_inner_seconds
        u_relax_inner = ts_inner if u_relax_inner is None else u_relax_inner
        v_relax_bound = _bcfg.sponge_v_timescale_bound_seconds
        v_relax_bound = ts_bound if v_relax_bound is None else v_relax_bound
        v_relax_inner = _bcfg.sponge_v_timescale_inner_seconds
        v_relax_inner = ts_inner if v_relax_inner is None else v_relax_inner
        sponge_width = _bcfg.sponge_width

        sides = tuple(str(s).upper() for s in (_bcfg.sides or ("N", "E", "S", "W")))
        side_set = set(sides)

        # OB_J*/OB_I* index tables.  Resolved from the policy the runtime sponges
        # apply (case_ob_policy: their ob_index_mode / ob_sides, else the boundary
        # fields), emitted for yaml boundary.sides (periodic-x global lat-lon
        # cases leave a side out).  The default
        # 'full_edge' run-length encodes to the historical `{nx}*1` etc., so this
        # stays a byte-for-byte match to legacy
        # experiments/alignment/mitgcm_export.py `_write_data_obcs`.
        ob_index_tables, ob_index_resolved = _ob_index_values(
            self._ctx, run_cfg, nx=nx, ny=ny, side_set=side_set
        )
        boundary_position_lines = "\n".join(
            _render_ob_index_line(*ob_index_resolved[letter])
            for letter, _keyword, _side in _OB_INDEX_SPECS
            if letter in ob_index_resolved
        )
        boundary_file_specs = (
            ("N", " OBNtFile='OBNt.bin',"),
            ("N", " OBNsFile='OBNs.bin',"),
            ("S", " OBStFile='OBSt.bin',"),
            ("S", " OBSsFile='OBSs.bin',"),
            ("E", " OBEtFile='OBEt.bin',"),
            ("E", " OBEsFile='OBEs.bin',"),
            ("W", " OBWtFile='OBWt.bin',"),
            ("W", " OBWsFile='OBWs.bin',"),
            ("N", " OBNvFile='OBNv.bin',"),
            ("S", " OBSvFile='OBSv.bin',"),
            ("E", " OBEuFile='OBEu.bin',"),
            ("W", " OBWuFile='OBWu.bin',"),
            ("N", " OBNuFile='OBNu.bin',"),
            ("S", " OBSuFile='OBSu.bin',"),
            ("E", " OBEvFile='OBEv.bin',"),
            ("W", " OBWvFile='OBWv.bin',"),
        )
        boundary_file_lines = "\n".join(
            line for side, line in boundary_file_specs if side in side_set
        )

        # OBCS_PARM01 logical switches (OBCSfixTopo / useOBCSbalance /
        # useOBCSsponge / useOBCSprescribe) come from the single-source table in
        # layer_d/_mit_switches.py, which resolves each one from the SAME run_cfg
        # the LOCOMgcm runtime reads. Preserve the configured boundary and
        # forcing conventions.
        # ``assert_switches_agree`` below re-parses the rendered text to prove
        # the file on disk still matches run_cfg.
        obcs_switch_lines = render_switch_lines(run_cfg, "data.obcs")

        # OBCS Tides namelist block
        tidal_lines = ""
        tidal_constituents = run_cfg.boundary.tidal_constituents
        # Prefer the per-constituent bins written by the stage1c mit_tidal lane
        # (mitgcm_tidal_binary → input/tides_raw/); fall back to the legacy
        # boundary.tidal_data_dir for pre-lane cases. This removes the hardcoded
        # data/tpxo9/obcs/{case} dependency while keeping the V0 + stacking +
        # big-endian export logic below unchanged.
        _lane_tides_raw = out_dir / "tides_raw"
        tidal_data_dir = (
            str(_lane_tides_raw) if _lane_tides_raw.exists()
            else run_cfg.boundary.tidal_data_dir
        )
        if tidal_constituents:
            tidal_lines += " useOBCStides=.TRUE.,\n"
            for i, comp in enumerate(tidal_constituents, 1):
                tidal_lines += f" OBCS_tidalPeriod({i})={float(comp['period']):.1f},\n"
            name_map = [
                ("N", "u", "OBN_uTidAmFile", "OBN_uTidPhFile", "uTidAm", "uTidPh"),
                ("N", "v", "OBN_vTidAmFile", "OBN_vTidPhFile", "vTidAm", "vTidPh"),
                ("S", "u", "OBS_uTidAmFile", "OBS_uTidPhFile", "uTidAm", "uTidPh"),
                ("S", "v", "OBS_vTidAmFile", "OBS_vTidPhFile", "vTidAm", "vTidPh"),
                ("E", "u", "OBE_uTidAmFile", "OBE_uTidPhFile", "uTidAm", "uTidPh"),
                ("E", "v", "OBE_vTidAmFile", "OBE_vTidPhFile", "vTidAm", "vTidPh"),
                ("W", "u", "OBW_uTidAmFile", "OBW_uTidPhFile", "uTidAm", "uTidPh"),
                ("W", "v", "OBW_vTidAmFile", "OBW_vTidPhFile", "vTidAm", "vTidPh"),
            ]
            # MITgcm reads one file per side/var containing ALL constituents stacked.
            # TPXO9 extraction produces per-constituent files; we must concatenate.
            #
            # V0 ASTRONOMICAL CORRECTION: TPXO9 phase files are referenced to a
            # Greenwich epoch (independent of model start_date). MITgcm has no
            # built-in V0 correction. LOCOMgcm's runtime tidal injection applies
            # V0(start_date) via boundaries/tidal._compute_v0_correction; we must
            # apply the SAME V0 here so MIT phase files are bit-equivalent.
            # Without this: LOCOMgcm uses V0-corrected phase, MIT uses raw → 30%
            # boundary v_N injection mismatch at sim 6h.
            tides_input_dir = out_dir / "tides"
            tides_input_dir.mkdir(parents=True, exist_ok=True)
            from pathlib import Path as _P
            import numpy as _np
            from defgcm.boundaries.tidal import tidal_phase_correction_seconds
            try:
                start_date_str = str(run_cfg.run.start_date)[:10]
            except AttributeError:
                start_date_str = "2019-01-01"  # safe fallback
            # Precompute V0 per constituent (seconds).  Skip when the phase data
            # already includes V0 (Gong native _Gph) -- matches the LOCOMgcm side
            # (enrich_manifest passes start_date=None) so both stay aligned and
            # neither double-counts.
            _skip_v0 = bool(getattr(run_cfg.boundary, "tidal_phase_includes_v0", False))
            v0_per_comp = [
                tidal_phase_correction_seconds(comp, None if _skip_v0 else start_date_str)
                for comp in tidal_constituents
            ]
            for side, uv, am_key, ph_key, am_pfx, ph_pfx in name_map:
                if side not in side_set:
                    continue
                am_stack, ph_stack = [], []
                for c_idx, comp in enumerate(tidal_constituents):
                    cname = comp["name"]
                    am_file = _P(tidal_data_dir) / f"OB{side}_{am_pfx}_{cname}.bin"
                    ph_file = _P(tidal_data_dir) / f"OB{side}_{ph_pfx}_{cname}.bin"
                    if am_file.exists() and ph_file.exists():
                        am_stack.append(_np.fromfile(str(am_file), dtype=_np.float64))
                        # Apply V0 correction: phase is in seconds (same unit as LOCOMgcm)
                        ph_raw = _np.fromfile(str(ph_file), dtype=_np.float64)
                        ph_corrected = ph_raw + v0_per_comp[c_idx]
                        ph_stack.append(ph_corrected)
                    elif am_file.exists() or ph_file.exists():
                        raise FileNotFoundError(
                            f"Tidal OB{side}_{uv}_{cname}: amplitude and phase files must both exist "
                            f"(am={am_file.exists()}, ph={ph_file.exists()})"
                        )
                if am_stack:
                    # Fortran column-major: all n_pts of comp1 first, then comp2.
                    # Big-endian: MIT compiles with -convert big_endian.
                    am_concat = _np.concatenate(am_stack)
                    am_out = tides_input_dir / f"OB{side}_{am_pfx}.bin"
                    am_concat.astype(">f8").tofile(str(am_out))
                    tidal_lines += f" {am_key}='tides/OB{side}_{am_pfx}.bin',\n"
                if ph_stack:
                    ph_concat = _np.concatenate(ph_stack)
                    ph_out = tides_input_dir / f"OB{side}_{ph_pfx}.bin"
                    ph_concat.astype(">f8").tofile(str(ph_out))
                    tidal_lines += f" {ph_key}='tides/OB{side}_{ph_pfx}.bin',\n"
            # Record V0 correction in namelist comment for traceability
            v0_summary = ", ".join(f"{c['name']}={v0:.1f}s" for c, v0 in zip(tidal_constituents, v0_per_comp))
            tidal_lines = f"# V0 correction applied for start_date={start_date_str}: {v0_summary}\n" + tidal_lines

        text = f"""\
# OBCS — generated by mitgcm_export.py
# Prescribed boundary condition for U/V/T/S with Sponge layer
 &OBCS_PARM01
{boundary_position_lines}
{obcs_switch_lines} OBCS_monitorFreq={obcs_period},
{boundary_file_lines}
{tidal_lines} &

 &OBCS_PARM02
 &

 &OBCS_PARM03
 spongeThickness = {sponge_width},
 Urelaxobcsinner = {u_relax_inner:.6g},
 Vrelaxobcsinner = {v_relax_inner:.6g},
 Urelaxobcsbound = {u_relax_bound:.6g},
 Vrelaxobcsbound = {v_relax_bound:.6g},
 &
"""
        assert_switches_agree(run_cfg, "data.obcs", text)
        _assert_ob_index_agreement(text, ob_index_resolved, ob_index_tables)
        _assert_obcs_sponge_consistency(
            run_cfg,
            u_bound=u_relax_bound, u_inner=u_relax_inner,
            v_bound=v_relax_bound, v_inner=v_relax_inner,
            sponge_width=sponge_width,
        )
        path = out_dir / "data.obcs"
        path.write_text(text)
        return ArtifactRef(
            name="mitgcm_obcs_namelist", path=path,
            artifact_kind=ArtifactKind.FILE_TEXT,
            bundle_hash="00000000", schema_version="0.1.0",
            extras={"input_dir": str(out_dir), "obcs_period_s": obcs_period,
                    "tidal_constituents": len(tidal_constituents)},
        )


@register_consumer("mitgcm_cpp_options")
class MitgcmCppOptionsAdapter:
    """Writes code/CPP_OPTIONS.h by wrapping legacy ``_write_cpp_options``.

    consumer_config (per-yaml-pipeline):
      extra_defines : list[str] — append ``#define <sym>`` to template
      extra_undefs  : list[str] — append ``#undef  <sym>`` to template

    Use to opt-in/-out of CPP symbols on a per-case basis without forking
    the template (e.g. ``EXACT_CONSERV`` for high-precision free-surface,
    ``ALLOW_DEBUG`` undef where the symbol breaks the host link).
    """

    capabilities = ConsumerCapabilities(
        accepts_location_kinds=tuple(LocationKind),
        required_convention=ConventionTag.OPAQUE,
        required_variables=(),
        artifact_kind=ArtifactKind.FILE_TEXT,
        artifact_schema_version="0.1.0",
    )

    def __init__(
        self,
        ctx: ConsumerContext,
        *,
        extra_defines: list[str] | None = None,
        extra_undefs: list[str] | None = None,
        **_: Any,
    ) -> None:
        self._ctx = ctx
        self._extra_defines = list(extra_defines or [])
        self._extra_undefs = list(extra_undefs or [])

    def consume(self, bundle: DataBundle) -> ArtifactRef:
        code_dir = self._ctx.output_dir / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        path = code_dir / "CPP_OPTIONS.h"
        params = _read_model_input_run_params(self._ctx)
        nonhydro = bool(_run_param(params, "nonhydrostatic", False))
        # MITgcm hides the whole implicit vertical advection code behind the CPP
        # macro INCLUDE_IMPLVERTADV_CODE; with the macro undefined and
        # *ImplVertAdv=.TRUE., model/src/config_check.F:205-246 reports each
        # switch and stops in CONFIG_CHECK.  The keyword and the macro that makes
        # it valid are therefore emitted from this one place.

        implvertadv = any(
            bool(_run_param(params, f"{leaf}_implicit_vertical_advection", False))
            for leaf in ("temperature", "salinity", "momentum")
        )
        cpp_defines = list(self._extra_defines)
        if implvertadv and "INCLUDE_IMPLVERTADV_CODE" not in cpp_defines:
            cpp_defines.append("INCLUDE_IMPLVERTADV_CODE")
        _legacy_export_module()._write_cpp_options(
            path,
            nonhydrostatic=nonhydro,
            extra_defines=cpp_defines,
            extra_undefs=self._extra_undefs,
        )
        momentum_advection_form = (
            momentum_advection_form_of(self._ctx.run_cfg) if self._ctx.run_cfg is not None
            else str(_run_param(params, "momentum_advection_form", "vector_invariant"))
        )
        enable_vertical_shear = bool(_run_param(params, "enable_vertical_shear", True))
        mom_vecinv_patched = False
        if momentum_advection_form == "vector_invariant" and not enable_vertical_shear:
            _legacy_export_module()._patch_disable_vertshear(code_dir)
            mom_vecinv_patched = True
        return ArtifactRef(
            name="mitgcm_cpp_options", path=path,
            artifact_kind=ArtifactKind.FILE_TEXT,
            bundle_hash=_bundle_hash(bundle), schema_version="0.1.0",
            extras={
                "nonhydrostatic": nonhydro,
                "extra_defines": cpp_defines,
                "implicit_vertical_advection_cpp": implvertadv,
                "extra_undefs": self._extra_undefs,
                "momentum_advection_form": momentum_advection_form,
                "enable_vertical_shear": enable_vertical_shear,
                "mom_vecinv_patched": mom_vecinv_patched,
                "wrapped": "_write_cpp_options",
            },
        )


@register_consumer("mitgcm_exf_namelist")
class MitgcmExfNamelistAdapter:
    """Writes input/data.exf — EXF in direct wind-stress mode.

    EXF reads pre-computed ustress.bin / vstress.bin (Large & Pond stress,
    produced by wind_to_stress operator + mitgcm_exf_binary consumer). Bulk
    formulae are disabled (see EXF_OPTIONS.h) so EXF only applies momentum
    forcing — no spurious heat/freshwater flux.

    The text is written by legacy ``mitgcm_export._write_data_exf``, so the two
    writers share one template; ``windstressmax`` is the clip LOCOMgcm's
    ``era5_wind_forcing`` applies (``_mit_switches.windstressmax_value``).

    Periods:
      stress_period_hours — interval between ustress/vstress records. ERA5 is
        natively 6-hourly, so 6.0 (set via consumer_config).
      repeatPeriod        — `obcs_repeat_period` = OBC time span from t0
        (legacy uses boundary_times[-1]); EXF cycles forcing past it.
      obcs?period         — OBC record interval (legacy obcs_period =
        boundary_times[1] - boundary_times[0]).

    primary_source must be the boundary bundle so the OBC time axis is
    available for repeatPeriod / &EXF_NML_OBCS.
    """

    capabilities = ConsumerCapabilities(
        accepts_location_kinds=tuple(LocationKind),
        required_convention=ConventionTag.OPAQUE,
        required_variables=(),
        artifact_kind=ArtifactKind.FILE_TEXT,
        artifact_schema_version="0.1.0",
    )

    def __init__(self, ctx: ConsumerContext, *,
                 stress_period_hours: float = 6.0,
                 **_: Any) -> None:
        self._ctx = ctx
        self._stress_period = float(stress_period_hours) * 3600.0

    def consume(self, bundle: DataBundle) -> ArtifactRef:
        run_cfg = self._ctx.run_cfg
        if run_cfg is None:
            raise ValueError("mitgcm_exf_namelist: ctx.run_cfg is None")
        sd = run_cfg.run.start_date
        date_int = sd.year * 10000 + sd.month * 100 + sd.day

        # OBC timing from the boundary bundle (legacy: obcs_period =
        # times[1]-times[0], obcs_repeat_period = times[-1]).
        obcs_period, repeat, _n = _extract_obcs_timing(bundle)
        exf_period = self._stress_period

        out_dir = self._ctx.output_dir / "input"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "data.exf"
        # One template for both writers; windstressmax comes from the LOCOMgcm wind
        # module (layer_d/_mit_switches.py) and is re-parsed below.
        _legacy_export_module()._write_data_exf(
            path,
            exf_period=exf_period,
            obcs_period=obcs_period,
            windstressmax=windstressmax_value(run_cfg),
            obcs_repeat_period=repeat,
            start_date_ymd=date_int,
        )
        assert_windstressmax_agrees(run_cfg, path.read_text())
        return ArtifactRef(
            name="mitgcm_exf_namelist", path=path,
            artifact_kind=ArtifactKind.FILE_TEXT,
            bundle_hash="00000000", schema_version="0.1.0",
            extras={"input_dir": str(out_dir),
                    "stress_period_s": self._stress_period,
                    "obcs_period_s": obcs_period,
                    "repeat_period_s": repeat},
        )


@register_consumer("mitgcm_exf_options")
class MitgcmExfOptionsAdapter:
    """Writes code/EXF_OPTIONS.h by wrapping legacy ``_write_exf_options``."""

    capabilities = ConsumerCapabilities(
        accepts_location_kinds=tuple(LocationKind),
        required_convention=ConventionTag.OPAQUE,
        required_variables=(),
        artifact_kind=ArtifactKind.FILE_TEXT,
        artifact_schema_version="0.1.0",
    )

    def __init__(self, ctx: ConsumerContext, **_: Any) -> None:
        self._ctx = ctx

    def consume(self, bundle: DataBundle) -> ArtifactRef:
        code_dir = self._ctx.output_dir / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        path = code_dir / "EXF_OPTIONS.h"
        _legacy_export_module()._write_exf_options(path)
        return ArtifactRef(
            name="mitgcm_exf_options", path=path,
            artifact_kind=ArtifactKind.FILE_TEXT,
            bundle_hash=_bundle_hash(bundle), schema_version="0.1.0",
            extras={"wrapped": "_write_exf_options"},
        )


@register_consumer("mitgcm_obcs_options")
class MitgcmObcsOptionsAdapter:
    """Writes code/OBCS_OPTIONS.h by wrapping legacy ``_write_obcs_options``."""

    capabilities = ConsumerCapabilities(
        accepts_location_kinds=tuple(LocationKind),
        required_convention=ConventionTag.OPAQUE,
        required_variables=(),
        artifact_kind=ArtifactKind.FILE_TEXT,
        artifact_schema_version="0.1.0",
    )

    def __init__(self, ctx: ConsumerContext, **_: Any) -> None:
        self._ctx = ctx

    def consume(self, bundle: DataBundle) -> ArtifactRef:
        code_dir = self._ctx.output_dir / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        path = code_dir / "OBCS_OPTIONS.h"
        sponge_width = _legacy_sponge_width(self._ctx)
        run_cfg = self._ctx.run_cfg
        enable_tides = bool(run_cfg.boundary.tidal_constituents) if run_cfg else False
        sides = tuple(str(s).upper() for s in (run_cfg.boundary.sides or ("N", "E", "S", "W"))) if run_cfg else ("N", "E", "S", "W")
        _legacy_export_module()._write_obcs_options(
            path,
            sponge_width=sponge_width,
            enable_tides=enable_tides,
            sides=sides,
        )
        return ArtifactRef(
            name="mitgcm_obcs_options", path=path,
            artifact_kind=ArtifactKind.FILE_TEXT,
            bundle_hash=_bundle_hash(bundle), schema_version="0.1.0",
            extras={"sponge_width": sponge_width, "enable_tides": enable_tides, "sides": sides},
        )


@register_consumer("mitgcm_cal_namelist")
class MitgcmCalNamelistAdapter:
    """Writes input/data.cal by wrapping legacy ``_write_data_cal``."""

    capabilities = ConsumerCapabilities(
        accepts_location_kinds=tuple(LocationKind),
        required_convention=ConventionTag.OPAQUE,
        required_variables=(),
        artifact_kind=ArtifactKind.FILE_TEXT,
        artifact_schema_version="0.1.0",
    )

    def __init__(self, ctx: ConsumerContext, **_: Any) -> None:
        self._ctx = ctx

    def consume(self, bundle: DataBundle) -> ArtifactRef:
        out_dir = self._ctx.output_dir / "input"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "data.cal"
        date_int = _legacy_start_date_ymd(self._ctx)
        _legacy_export_module()._write_data_cal(path, start_date_ymd=date_int)
        return ArtifactRef(
            name="mitgcm_cal_namelist", path=path,
            artifact_kind=ArtifactKind.FILE_TEXT,
            bundle_hash=_bundle_hash(bundle), schema_version="0.1.0",
            extras={"input_dir": str(out_dir), "start_date_ymd": date_int, "wrapped": "_write_data_cal"},
        )


@register_consumer("mitgcm_eedata")
class MitgcmEedataAdapter:
    """Writes input/eedata by wrapping legacy ``_write_eedata``."""

    capabilities = ConsumerCapabilities(
        accepts_location_kinds=tuple(LocationKind),
        required_convention=ConventionTag.OPAQUE,
        required_variables=(),
        artifact_kind=ArtifactKind.FILE_TEXT,
        artifact_schema_version="0.1.0",
    )

    def __init__(self, ctx: ConsumerContext, **_: Any) -> None:
        self._ctx = ctx

    def consume(self, bundle: DataBundle) -> ArtifactRef:
        out_dir = self._ctx.output_dir / "input"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "eedata"
        _legacy_export_module()._write_eedata(path)
        return ArtifactRef(
            name="mitgcm_eedata",
            path=path,
            artifact_kind=ArtifactKind.FILE_TEXT,
            bundle_hash=_bundle_hash(bundle),
            schema_version="0.1.0",
            extras={"input_dir": str(out_dir), "wrapped": "_write_eedata"},
        )
