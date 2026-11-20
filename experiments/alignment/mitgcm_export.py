#!/usr/bin/env python3
"""Export LOCOMgcm prepared data to MITgcm binary format and configuration files.

Creates a self-contained MITgcm case directory with:
  code/   — SIZE.h, packages.conf, CPP_OPTIONS.h, OBCS_OPTIONS.h, EXF_OPTIONS.h
  input/  — data, data.pkg, data.obcs, data.exf, data.cal, eedata, binary files
  build/  — (empty, created for genmake2)
  run/    — (empty, created for execution)

Usage from compare_hycom.py:
    from mitgcm_export import export_mitgcm_case
    case_dir = export_mitgcm_case(output_dir / "mitgcm", ...)
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from defgcm.external_paths import resolve
from typing import Any

import numpy as np
import torch

from defgcm.dycore_parameters import NonhydrostaticParams
from defgcm.dynamics.era5_wind_forcing import WINDSTRESSMAX_EXF_DEFAULT
from defgcm.geometry import MaskSet
from defgcm.io.reference_profile import compute_reference_tracer_profile
from defgcm.io.runner.config.vertical_diffusion import (
    DEFAULT_DIFFUSIVITY_KR, DEFAULT_IMPLICIT_DIFFUSION,
)
from defgcm.io.runner.layer_d._mit_switches import (
    _namelist_real,
    render_windstressmax,
    windstressmax_namelist_value,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MITGCM_HFAC_MIN = 0.3
MITGCM_HFAC_MIN_DZ = 5.0
MITGCM_START_DATE_YMD = 20190101
_MITGCM_OBCS_FIELD_SPECS = (
    ("temp", "temp", "t"),
    ("salt", "salt", "s"),
    ("u", "u", "u"),
    ("v", "v", "v"),
)


# ──────────────────────────────────────────────────────────────────────
# Binary I/O
# ──────────────────────────────────────────────────────────────────────

def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _write_binary(path: Path, data: np.ndarray | torch.Tensor) -> None:
    """Write a tensor/array as big-endian float64 flat binary (MITgcm convention)."""
    arr = np.asarray(_to_numpy(data), dtype=np.float64)
    arr_be = arr.astype(">f8")  # big-endian float64
    arr_be.tofile(str(path))


def read_binary_checked(path: Path, shape: tuple[int, ...], dtype: str = ">f8") -> np.ndarray:
    data = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if data.size != expected:
        raise ValueError(f"{path}: expected {expected} values, found {data.size}")
    return data.reshape(shape).astype(np.float64)


def project_state_to_mitgcm_initial_fields(
    state: Any,
    *,
    nx: int,
    ny: int,
    nz: int,
) -> dict[str, np.ndarray]:
    eta = _to_numpy(state.eta.data)
    temp = _to_numpy(state.temp.data)
    salt = _to_numpy(state.salt.data)
    u = _to_numpy(state.u.data)
    v = _to_numpy(state.v.data)

    if eta.ndim == 3:
        eta = eta[0]
    if temp.ndim == 4:
        temp = temp[0]
    if salt.ndim == 4:
        salt = salt[0]
    if u.ndim == 4:
        u = u[0]
    if v.ndim == 4:
        v = v[0]

    # LOCOMgcm's u_face is shape [nz, ny, nx+1] with u_face[i] at the WEST
    # face of cell i (for i < nx) and u_face[nx] at the east domain edge.
    # MITgcm's uVelInitFile.bin stores shape [nz, ny, nx] where U(i,j,k) IS
    # the velocity at the west face of cell (i,j,k) — same convention as
    # LOCOMgcm's u_face[:, :, :nx].  Drop the last face, DO NOT average.
    # (Averaging 0.5*(u_face[:-1] + u_face[1:]) would double-
    # interpolate and halve values at wet/dry interfaces, producing a
    # step-0 U/V offset of up to 1 m/s at boundary partial cells.)
    if u.shape[-1] == nx + 1:
        u = u[..., :nx]
    if v.shape[-2] == ny + 1:
        v = v[..., :ny, :]

    return {
        "eta": np.nan_to_num(np.asarray(eta, dtype=np.float64), nan=0.0),
        "temp": np.nan_to_num(np.asarray(temp, dtype=np.float64), nan=0.0),
        "salt": np.nan_to_num(np.asarray(salt, dtype=np.float64), nan=0.0),
        "u": np.nan_to_num(np.asarray(u, dtype=np.float64), nan=0.0),
        "v": np.nan_to_num(np.asarray(v, dtype=np.float64), nan=0.0),
    }


def project_bathymetry_to_mitgcm(bathymetry: np.ndarray | torch.Tensor) -> np.ndarray:
    bathy = _to_numpy(bathymetry)
    if bathy.ndim > 2:
        bathy = bathy.squeeze()
    return np.where(bathy > 0, -np.asarray(bathy, dtype=np.float64), 0.0)


def project_boundary_source_to_mitgcm_files(boundary_source: Any) -> dict[str, np.ndarray]:
    exports: dict[str, np.ndarray] = {}
    for _, field_name, suffix in _MITGCM_OBCS_FIELD_SPECS:
        field_obj = getattr(boundary_source, field_name, None)
        if field_obj is None:
            continue
        for side in ["west", "east", "south", "north"]:
            side_data = getattr(field_obj, side, None)
            if side_data is None:
                continue
            side_prefix = side[0].upper()
            exports[f"OB{side_prefix}{suffix}.bin"] = np.nan_to_num(
                np.asarray(_to_numpy(side_data), dtype=np.float64),
                nan=0.0,
            )
    return exports


def project_era5_source_to_mitgcm_files(era5_source: Any) -> dict[str, np.ndarray]:
    return {
        "ustress.bin": np.asarray(_to_numpy(era5_source.tau_x), dtype=np.float64),
        "vstress.bin": np.asarray(_to_numpy(era5_source.tau_y), dtype=np.float64),
    }


def reconstruct_hfac_from_bathy(
    bathymetry: np.ndarray | torch.Tensor,
    *,
    interface_depth: np.ndarray | torch.Tensor,
    boundary_topology: Any,
    hfac_min: float = MITGCM_HFAC_MIN,
    hfac_min_dz: float = MITGCM_HFAC_MIN_DZ,
) -> dict[str, np.ndarray]:
    bathy_np = np.asarray(_to_numpy(bathymetry), dtype=np.float64)
    if bathy_np.ndim > 2:
        bathy_np = bathy_np.squeeze()
    positive_bathy = np.where(bathy_np < 0.0, -bathy_np, np.where(bathy_np > 0.0, bathy_np, 0.0))
    mask_set = MaskSet.from_bathymetry_3d(
        torch.as_tensor(positive_bathy, dtype=torch.float64),
        torch.as_tensor(_to_numpy(interface_depth), dtype=torch.float64),
        device="cpu",
        hfac_min=hfac_min,
        hfac_min_dz=hfac_min_dz,
        boundary_topology=boundary_topology,
    )
    return {
        "hFacC": _to_numpy(mask_set.hFacC),
        "hFacW": _to_numpy(mask_set.hFacW),
        "hFacS": _to_numpy(mask_set.hFacS),
        "maskC": _to_numpy(mask_set.center).astype(bool),
        "maskW": _to_numpy(mask_set.u).astype(bool),
        "maskS": _to_numpy(mask_set.v).astype(bool),
    }


# ──────────────────────────────────────────────────────────────────────
# Code directory: SIZE.h, packages.conf, CPP_OPTIONS.h, OBCS_OPTIONS.h
# ──────────────────────────────────────────────────────────────────────

def _write_size_h(
    path: Path,
    *,
    nx: int,
    ny: int,
    nz: int,
    n_px: int = 1,
    n_py: int = 1,
    sNx: int | None = None,
    sNy: int | None = None,
    ol_x: int = 4,
    ol_y: int = 4,
) -> None:
    """Generate SIZE.h for an arbitrary nPx × nPy MPI decomposition.

    sNx / sNy default to nx // n_px and ny // n_py; integer-divisibility is
    asserted (RunConfig.__post_init__ enforces this from yaml too, but
    catch local-call abuse).
    """
    if sNx is None:
        if nx % n_px != 0:
            raise ValueError(f"_write_size_h: nx={nx} not divisible by n_px={n_px}")
        sNx = nx // n_px
    if sNy is None:
        if ny % n_py != 0:
            raise ValueError(f"_write_size_h: ny={ny} not divisible by n_py={n_py}")
        sNy = ny // n_py
    path.write_text(f"""\
C     SIZE.h — nPx={n_px} nPy={n_py} (generated by mitgcm_export.py)
      INTEGER sNx
      INTEGER sNy
      INTEGER OLx
      INTEGER OLy
      INTEGER nSx
      INTEGER nSy
      INTEGER nPx
      INTEGER nPy
      INTEGER Nx
      INTEGER Ny
      INTEGER Nr
      PARAMETER (
     &           sNx = {sNx:4d},
     &           sNy = {sNy:4d},
     &           OLx = {ol_x:4d},
     &           OLy = {ol_y:4d},
     &           nSx =    1,
     &           nSy =    1,
     &           nPx = {n_px:4d},
     &           nPy = {n_py:4d},
     &           Nx  = sNx*nSx*nPx,
     &           Ny  = sNy*nSy*nPy,
     &           Nr  = {nz:4d})

      INTEGER MAX_OLX
      INTEGER MAX_OLY
      PARAMETER ( MAX_OLX = OLx,
     &            MAX_OLY = OLy )
""")


def _write_packages_conf(
    path: Path,
    *,
    include_exf: bool = True,
    include_kpp: bool = False,
    diagnostics: bool = False,
) -> None:
    """Generate packages.conf for OBCS (+ optional EXF/KPP).

    ``include_exf`` defaults True so every existing case compiles exf exactly
    as before. No-wind cases (e.g. Gong ISWFM, sponge+tides only) pass False to
    drop exf from the build — paired with ``use_exf=False`` in data.pkg.
    """
    lines = ["#-- packages to compile (generated by mitgcm_export.py)", "oceanic", "obcs"]
    if include_kpp:
        lines.append("kpp")
    if include_exf:
        lines.append("exf")
    if diagnostics:
        lines.append("diagnostics")
    path.write_text("\n".join(lines) + "\n")


def _write_cpp_options(
    path: Path,
    *,
    nonhydrostatic: bool = False,
    extra_defines: list[str] | None = None,
    extra_undefs: list[str] | None = None,
) -> None:
    """Generate minimal CPP_OPTIONS.h for the selected ocean build.

    extra_defines / extra_undefs are emitted after the canonical block so
    case-yaml can opt-in/-out of symbols the template doesn't know about
    (e.g. ``EXACT_CONSERV`` for high-precision free-surface, or
    ``ALLOW_DEBUG`` undef for builds without the required debug helpers).
    Survives stage1c rerun without external patching — the yaml is the
    source of truth.
    """
    nh_line = "#define ALLOW_NONHYDROSTATIC" if nonhydrostatic else "#undef  ALLOW_NONHYDROSTATIC"
    nh_description = "Nonhydrostatic" if nonhydrostatic else "Hydrostatic"

    extras_block = ""
    extra_defines = extra_defines or []
    extra_undefs = extra_undefs or []
    if extra_defines or extra_undefs:
        lines = ["", "C-- Case-yaml extras (extra_defines / extra_undefs)"]
        for sym in extra_undefs:
            lines.append(f"#undef  {sym}")
        for sym in extra_defines:
            lines.append(f"#define {sym}")
        extras_block = "\n".join(lines) + "\n"

    path.write_text(f"""\
#ifndef CPP_OPTIONS_H
#define CPP_OPTIONS_H

C     CPP_OPTIONS.h — generated by mitgcm_export.py
C     Minimal {nh_description} ocean configuration.

#define INCLUDE_PHIHYD_CALCULATION_CODE
#define INCLUDE_CONVECT_CALL
#define INCLUDE_CALC_DIFFUSIVITY_CALL
{nh_line}
#undef  NONLIN_FRSURF
#undef  ALLOW_ADDFLUID
#define ATMOSPHERIC_LOADING
#undef  EXCLUDE_FFIELDS_LOAD
{extras_block}
C-- Execution environment
#include "CPP_EEOPTIONS.h"

#endif /* CPP_OPTIONS_H */
""")


def _resolve_nonhydrostatic_export_controls(
    nonhydrostatic: bool | NonhydrostaticParams,
    cg3d_max_iters: int | None,
    cg3d_target_residual: float | None,
) -> tuple[bool, int, float, float, float]:
    """Resolve public export NH controls into MIT namelist/header values."""
    if isinstance(nonhydrostatic, NonhydrostaticParams):
        enabled = nonhydrostatic.nonhydrostatic
        resolved_max_iters = (
            nonhydrostatic.cg3d_max_iters
            if cg3d_max_iters is None
            else cg3d_max_iters
        )
        resolved_target_residual = (
            nonhydrostatic.cg3d_target_residual
            if cg3d_target_residual is None
            else cg3d_target_residual
        )
        resolved_target_res_w_unit = nonhydrostatic.cg3d_target_res_w_unit
        resolved_nh_am2 = nonhydrostatic.nh_am2
    else:
        enabled = bool(nonhydrostatic)
        resolved_max_iters = 150 if cg3d_max_iters is None else cg3d_max_iters
        resolved_target_residual = (
            1e-7 if cg3d_target_residual is None else cg3d_target_residual
        )
        resolved_target_res_w_unit = -1.0
        resolved_nh_am2 = 1.0

    return (
        enabled,
        resolved_max_iters,
        resolved_target_residual,
        resolved_target_res_w_unit,
        resolved_nh_am2,
    )


def _patch_disable_vertshear(code_dir: Path, mitgcm_root: Path | None = None) -> None:
    """Copy mom_vecinv.F into code/ with vertical shear calls disabled.

    Wraps the MOM_VI_U/V_VERTSHEAR calls and their gU/gV accumulation
    in an ``#ifdef ALLOW_MOM_VERTSHEAR`` block.  Since the CPP_OPTIONS.h
    we generate does not define that macro, the calls are compiled out.
    """
    src = resolve("mitgcm_root", mitgcm_root) / "pkg" / "mom_vecinv" / "mom_vecinv.F"
    if not src.exists():
        raise FileNotFoundError(f"MITgcm mom_vecinv.F not found: {src}")
    text = src.read_text()
    old_block = """\
C--   Vertical shear terms (-w*du/dr & -w*dv/dr)
       IF ( .NOT. momImplVertAdv ) THEN"""
    new_block = """\
C--   Vertical shear terms (-w*du/dr & -w*dv/dr)
C--   DISABLED by mitgcm_export.py: LOCOMgcm does not yet have a validated
C--   vertical shear implementation; skip to keep both models aligned.
       IF ( .FALSE. .AND. .NOT. momImplVertAdv ) THEN"""
    if old_block not in text:
        raise RuntimeError("Could not find vertical shear block in MITgcm mom_vecinv.F")
    text = text.replace(old_block, new_block, 1)
    (code_dir / "mom_vecinv.F").write_text(text)


def _write_obcs_options(
    path: Path,
    sponge_width: int,
    enable_tides: bool = False,
    sides: tuple[str, ...] | list[str] | None = None,
) -> None:
    """Generate OBCS_OPTIONS.h with Orlanski + sponge + prescribe."""
    sponge_macro = "#define ALLOW_OBCS_SPONGE" if sponge_width > 0 else "#undef ALLOW_OBCS_SPONGE"
    tide_macro = "#define ALLOW_OBCS_TIDES" if enable_tides else "#undef ALLOW_OBCS_TIDES"
    side_set = {str(side).upper() for side in (sides or ("N", "S", "E", "W"))}
    side_macros = "\n".join(
        f"#define ALLOW_OBCS_{name}"
        for side, name in (
            ("N", "NORTH"),
            ("S", "SOUTH"),
            ("E", "EAST"),
            ("W", "WEST"),
        )
        if side in side_set
    )
    path.write_text(f"""\
C     OBCS_OPTIONS.h — generated by mitgcm_export.py
C     Orlanski radiation + sponge + prescribe on configured sides.

#ifndef OBCS_OPTIONS_H
#define OBCS_OPTIONS_H
#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

#ifdef ALLOW_OBCS

{side_macros}

C  Orlanski radiation BC
#undef ALLOW_ORLANSKI

C  prescribe external boundary values from files
#define ALLOW_OBCS_PRESCRIBE

C  sponge layer relaxation
{sponge_macro}

C  balance barotropic velocity
#define ALLOW_OBCS_BALANCE

C  barotropic tidal velocity overlay
{tide_macro}

C  Stevens (1990) boundary condition
#undef ALLOW_OBCS_STEVENS

#endif /* ALLOW_OBCS */
#endif /* OBCS_OPTIONS_H */
""")


def _write_exf_options(path: Path) -> None:
    """Generate EXF_OPTIONS.h override — disable bulk formulae.

    MITgcm's default EXF_OPTIONS.h defines ALLOW_ATM_TEMP and
    ALLOW_BULKFORMULAE, causing the EXF package to compute surface
    heat flux via bulk formulae using atmospheric temperature, humidity,
    wind, etc. Our experiments provide ONLY wind stress (ustress.bin,
    vstress.bin) — no atmospheric temperature or humidity files. With
    bulk formulae enabled, MITgcm falls back to default atmospheric
    values (atemp=0K, aqh=0, etc.) producing a nonsense heat flux of
    ~1000-1600 W/m² cooling the ocean surface by ~2 K over 7 days.

    Undefining these flags ensures EXF only processes the wind stress
    files; Qnet / EmPmR / saltFlux remain zero (matching LOCOMgcm,
    which has no surface heat flux module).
    """
    path.write_text("""\
C     EXF_OPTIONS.h — generated by mitgcm_export.py
C     Override to disable bulk formulae. Our experiments provide only
C     wind stress; bulk formulae with default zero atmosphere produces
C     bogus ~1650 W/m² surface cooling and corrupts the alignment.

#ifndef EXF_OPTIONS_H
#define EXF_OPTIONS_H
#include "PACKAGES_CONFIG.h"
#include "CPP_OPTIONS.h"

#ifdef ALLOW_EXF

C     Disable bulk formulae and atmospheric temperature loading.
C     With these undef'd, EXF only processes the wind stress files
C     and leaves Qnet / EmPmR / saltFlux at zero.
#undef  ALLOW_ATM_TEMP
#undef  ALLOW_ATM_WIND
#undef  ALLOW_BULKFORMULAE
#undef  ALLOW_DOWNWARD_RADIATION

C     Disable other climatological forcing options we don't use
#undef  ALLOW_CLIMTEMP_RELAXATION
#undef  ALLOW_CLIMSALT_RELAXATION
#undef  ALLOW_CLIMSST_RELAXATION
#undef  ALLOW_CLIMSSS_RELAXATION

C     Disable seaice coupling / shortwave
#undef  ALLOW_ZENITHANGLE
#undef  ALLOW_RUNOFF
#undef  SHORTWAVE_HEATING

C     EXF interpolation (not needed — forcing is on the model grid)
#undef  USE_EXF_INTERPOLATION

#endif /* ALLOW_EXF */
#endif /* EXF_OPTIONS_H */
""")


# ──────────────────────────────────────────────────────────────────────
# Input directory: namelist configuration files
# ──────────────────────────────────────────────────────────────────────

#: MITgcm copies every parameter-file line into a CHARACTER*(MAX_LEN_PREC)
#: record before the namelist read (eesupp/src/open_copy_data_file.F:46,135),
#: and MAX_LEN_PREC = 200 (eesupp/inc/EEPARAMS.h:27); characters past column
#: 200 are dropped without a message.
MITGCM_MAX_LEN_PREC = 200


def _format_value_list(keyword: str, values: list[str]) -> str:
    """Render ``keyword= v1, v2, ...,`` five values per line.

    A namelist value list may continue over several lines (e.g.
    references/MITgcm/verification/exp2/input.rigidLid/data), and it has to,
    because of MITGCM_MAX_LEN_PREC. Truncated delZ lists cause missing-level
    errors. Truncated tRef and sRef lists can silently repeat values because
    load_ref_files.F fills every unset level from the one above.
    """
    vals_per_line = 5
    indent = "       "  # 7-space continuation indent
    chunks = [
        ", ".join(values[i:i + vals_per_line])
        for i in range(0, len(values), vals_per_line)
    ]
    return f" {keyword}= " + (",\n" + indent).join(chunks) + ","


def _format_delz(nz: int, depth_m: float, layer_thickness: Any | None = None) -> str:
    """Format delZ entries for uniform or explicit non-uniform vertical layers."""
    if layer_thickness is not None:
        dz = np.asarray(layer_thickness, dtype=np.float64)
        if dz.ndim != 1:
            raise ValueError("layer_thickness must be 1-D")
        if dz.size != nz:
            raise ValueError("layer_thickness length must equal nz")
        if not np.isfinite(dz).all():
            raise ValueError("layer_thickness values must be finite")
        if not (dz > 0.0).all():
            raise ValueError("layer_thickness values must be strictly positive")
        if abs(float(dz.sum()) - float(depth_m)) >= 1.0e-6:
            raise ValueError("layer_thickness sum must match depth_m within 1e-6")
        return _format_value_list("delZ", [_exact(v, f"{float(v):.10f}") for v in dz])
    dz = depth_m / nz
    return f" delZ= {nz}*{_exact(dz, f'{dz:.4f}')},"


def _format_tref(nz: int, init_temp: np.ndarray) -> str:
    """Format tRef from the horizontally-averaged initial temperature profile."""
    field = torch.as_tensor(init_temp) if isinstance(init_temp, torch.Tensor) else np.asarray(init_temp)
    if field.ndim == 3:
        profile = compute_reference_tracer_profile(field)
    elif field.ndim == 1:
        profile = np.asarray(field, dtype=np.float64)
    else:
        profile = np.full(nz, 20.0)
    return _format_value_list("tRef", [f"{float(v):.17g}" for v in profile[:nz]])


def _format_sref(nz: int, init_salt: np.ndarray) -> str:
    """Format sRef from the horizontally-averaged initial salinity profile."""
    field = torch.as_tensor(init_salt) if isinstance(init_salt, torch.Tensor) else np.asarray(init_salt)
    if field.ndim == 3:
        profile = compute_reference_tracer_profile(field)
        profile = [35.0 if value == 0.0 else value for value in profile]
    elif field.ndim == 1:
        profile = np.asarray(field, dtype=np.float64)
    else:
        profile = np.full(nz, 35.0)
    return _format_value_list("sRef", [f"{float(v):.17g}" for v in profile[:nz]])


def _format_vert_adv_scheme(
    tempVertAdvScheme: int | None,
    saltVertAdvScheme: int | None,
) -> str:
    """Emit explicit ``tempVertAdvScheme`` / ``saltVertAdvScheme`` namelist
    lines.  MITgcm ``ini_parms.F:509-510`` defaults vertical advection scheme
    to the horizontal scheme when not set; for DST3-FL (33) horizontal
    paired with LOCOMgcm's upwind R-pass, we MUST emit explicit
    ``tempVertAdvScheme=1`` / ``saltVertAdvScheme=1`` to prevent MIT from
    silently running DST3-FL vertical advection while LOCOMgcm runs upwind.
    Returns empty string when both args are None (legacy behavior).
    """
    lines = []
    if tempVertAdvScheme is not None:
        lines.append(f" tempVertAdvScheme={tempVertAdvScheme},")
    if saltVertAdvScheme is not None:
        lines.append(f" saltVertAdvScheme={saltVertAdvScheme},")
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _require_exact_bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool, got {type(value).__name__}")
    return value


def _exact(value: float, text: str) -> str:
    """``text`` when it reads back as exactly ``value``, else the shortest exact text.

    Keeps the template's spelling of a value it renders exactly (so existing
    namelists stay byte-identical) and never writes a rounded value.
    """
    value = float(value)
    probe = text.strip().upper().replace("D", "E")
    try:
        if float(probe) == value:
            return text
    except ValueError:
        pass
    return _namelist_real(value)


def _e_text(value: float) -> str:
    """The template's ``%.15G`` spelling with a dotted mantissa (``1.E-13``)."""
    text = f"{float(value):.15G}"
    if "E" in text:
        mantissa, exponent = text.split("E", 1)
        if "." not in mantissa:
            mantissa = f"{mantissa}."
        return f"{mantissa}E{exponent}"
    return text if "." in text else f"{text}.0"


def _write_data(
    path: Path,
    *,
    nx: int,
    ny: int,
    nz: int,
    resolution_deg: float,
    depth_m: float,
    dt_seconds: float,
    total_seconds: float,
    layer_thickness: Any | None = None,
    dt_mom_seconds: float | None = None,
    dt_tracer_seconds: float | None = None,
    dt_freesurf_seconds: float | None = None,
    dt_clock_seconds: float | None = None,
    viscosity: float,
    viscA4: float,
    # diffKhT and diffKhS as LOCOMgcm's horizontal tracer diffusion runs them
    # (layer_d/_mit_switches.tracer_harmonic_diffusivity).
    diffusivity_t: float,
    diffusivity_s: float,
    xg_origin: float,
    yg_origin: float,
    dump_freq: float,
    init_temp: np.ndarray,
    init_salt: np.ndarray,
    temperature_implicit_vertical_advection: bool,
    salinity_implicit_vertical_advection: bool,
    momentum_implicit_vertical_advection: bool,
    tempAdvScheme: int | None = 33,
    saltAdvScheme: int | None = 33,
    tempVertAdvScheme: int | None = None,
    saltVertAdvScheme: int | None = None,
    # PARM01 equation of state and constants as LOCOMgcm holds them
    # (layer_d/_mit_switches.py eos_namelist_values, rho_const_value,
    # gravity_value).  ``eos`` has eosType, and for LINEAR, the only type
    # that reads them (find_rho.F:98-108), tAlpha, sBeta, rhoNil and the
    # module's tRef and sRef level lists.  A LINEAR ``eos`` without tRef/sRef
    # (compare_hycom, whose model takes the same wet-cell mean of its initial
    # field) writes the initial-field mean.
    eos: dict[str, Any],
    rho_const: float,
    gravity: float,
    use_cartesian: bool = False,
    dx_m: float | None = None,
    dy_m: float | None = None,
    # Cartesian f0 and beta [s-1, m-1 s-1] of the LOCOMgcm Coriolis parameter.
    f0: float | None = None,
    beta: float | None = None,
    implicit_diffusion: bool = DEFAULT_IMPLICIT_DIFFUSION,
    tracer_vertical_diffusivity_kr: float = DEFAULT_DIFFUSIVITY_KR,
    implicit_viscosity: bool = True,
    no_slip_sides: bool = False,
    no_slip_bottom: bool = False,
    # Bottom drag coefficients (mom_u_botdrag_coeff.F).  The runner consumer
    # passes both from the LOCOMgcm bottom_drag module; the defaults only keep legacy
    # direct callers byte-identical (quadratic 0.0025 was the old template constant;
    # linear was never written).
    bottom_drag_linear: float = 0.0,
    bottom_drag_quadratic: float = 0.0025,
    # viscAz: the vertical viscosity LOCOMgcm runs (the stepper's implicit_viscosity).
    vertical_viscosity: float,
    hfac_min: float = MITGCM_HFAC_MIN,
    hfac_min_dz: float = MITGCM_HFAC_MIN_DZ,
    lin_fs_conserve_tr: bool = False,
    ke_scheme: int = 0,
    viscosity_scheme: str = "constant",
    visc_c2_leith: float = 0.0,
    visc_c2_smag: float = 0.0,
    momentum_advection_form: str = "vector_invariant",
    nonhydrostatic: bool = False,
    cg3d_max_iters: int = 150,
    cg3d_target_residual: float = 1e-7,
    cg3d_target_res_w_unit: float = -1.0,
    nh_am2: float = 1.0,
    # PARM03 Adams-Bashforth forcing placement, written as MITgcm's two integers
    # rather than via the ``forcing_In_AB`` logical this template once used.
    # That logical resolves BOTH at once (ini_parms.F:1095-1101), so
    # it cannot express LOCOMgcm's configuration, whose momentum side is a
    # per-case declaration (stepper.mom_forcing_out_ab) while its tracer side is
    # hardcoded post-AB2.  All three names are PARM03 namelist variables
    # (ini_parms.F:279), so writing the integers costs nothing.  The defaults
    # reproduce the retired ``forcing_In_AB=.FALSE.`` exactly; the v2 consumer
    # always passes values resolved from run_cfg (layer_d/_mit_switches.py) and
    # re-parses the written file to check them.
    mom_forcing_out_ab: int = 1,
    trac_forcing_out_ab: int = 1,
    # PARM01 exactConserv / staggerTimeStep / tempStepping / saltStepping.  The
    # v2 consumer passes the values LOCOMgcm runs with (the stepper kwargs and
    # the tracer modules' tracer_names, layer_d/_mit_switches.py) and re-parses
    # the written file; the .TRUE. defaults are the old hardcoded template values
    # and keep legacy direct callers byte-identical.
    exact_conserv: bool = True,
    stagger_time_step: bool = True,
    temp_stepping: bool = True,
    salt_stepping: bool = True,
    # Extra PARM01 viscosity lines (sideDragFactor and the mom_calc_visc.F
    # variable-viscosity keywords), rendered from LOCOMgcm's
    # horizontal_viscosity module by layer_d/_mit_switches.render_viscosity_lines
    # and placed after viscC2smag.  Empty for legacy direct callers.
    viscosity_extra_lines: str = "",
    # PARM01 diffK4T and diffK4S, the biharmonic temperature and salinity
    # diffusivities [m^4/s]: the v2 consumer passes the horizontal_tracer_diffusion
    # coefficient of the modules that list each tracer
    # (layer_d/_mit_switches.tracer_biharmonic_diffusivity).  0 omits that key,
    # MITgcm's default (set_defaults.F:155-156), so a case without biharmonic
    # diffusion keeps a line-for-line identical ``data``.
    diffusivity_k4_temp: float = 0.0,
    diffusivity_k4_salt: float = 0.0,
    # PARM03 abEps, the Adams-Bashforth weight abFac = 0.5 + abEps
    # (adams_bashforth2.F:64).  The v2 consumer passes the stepper's ab_eps
    # (layer_d/_mit_switches.ab_eps_value) and re-parses the written file; the
    # 0.1 default is the old hardcoded template value and keeps
    # legacy direct callers byte-identical.
    ab_eps: float = 0.1,
    # The keys below were template constants;
    # the v2 consumer passes the model's values (layer_d/_mit_switches.py
    # DATA_KEYS) and re-parses the file.  The defaults keep legacy direct
    # callers byte-identical.
    implic_surf_press: float = 1.0,
    implic_div_2d_flow: float = 1.0,
    mom_stepping: bool = True,
    multi_dim_advection: bool = True,
    cg2d: dict[str, float] | None = None,
    c_adj_freq: float = 0.0,
) -> None:
    """Generate the main 'data' namelist file (PARM01-05)."""
    cg2d = {"cg2dMaxIters": 1000, "cg2dTargetResidual": 1e-13} if cg2d is None else cg2d
    exact_conserv = _require_exact_bool(exact_conserv, name="exact_conserv")
    stagger_time_step = _require_exact_bool(stagger_time_step, name="stagger_time_step")
    temp_stepping = _require_exact_bool(temp_stepping, name="temp_stepping")
    salt_stepping = _require_exact_bool(salt_stepping, name="salt_stepping")
    temperature_implicit_vertical_advection = _require_exact_bool(
        temperature_implicit_vertical_advection,
        name="temperature_implicit_vertical_advection",
    )
    salinity_implicit_vertical_advection = _require_exact_bool(
        salinity_implicit_vertical_advection,
        name="salinity_implicit_vertical_advection",
    )
    momentum_implicit_vertical_advection = _require_exact_bool(
        momentum_implicit_vertical_advection,
        name="momentum_implicit_vertical_advection",
    )
    if "tRef" in eos:
        tref_line = _format_value_list("tRef", [f"{float(v):.17g}" for v in eos["tRef"]])
        sref_line = _format_value_list("sRef", [f"{float(v):.17g}" for v in eos["sRef"]])
    else:
        tref_line = _format_tref(nz, init_temp)
        sref_line = _format_sref(nz, init_salt)
    delz_line = _format_delz(nz, depth_m, layer_thickness)
    dt_mom = dt_seconds if dt_mom_seconds is None else dt_mom_seconds
    dt_tracer = dt_seconds if dt_tracer_seconds is None else dt_tracer_seconds
    dt_freesurf = dt_seconds if dt_freesurf_seconds is None else dt_freesurf_seconds
    dt_clock = dt_seconds if dt_clock_seconds is None else dt_clock_seconds
    # MITgcm monitor output: every 100 steps
    monitor_freq = dt_clock * 100

    # EOS-specific lines
    eos_lines = f" eosType='{eos['eosType']}',"
    rho_nil_line = ""
    if eos["eosType"] == "LINEAR":
        eos_lines += f"""
 tAlpha={_namelist_real(eos['tAlpha'])},
 sBeta={_namelist_real(eos['sBeta'])},"""
        rho_nil_line = f" rhoNil={_namelist_real(eos['rhoNil'])},\n"
    if "selectP_inEOS_Zc" in eos and eos["selectP_inEOS_Zc"] != 2:
        eos_lines += f"\n selectP_inEOS_Zc={int(eos['selectP_inEOS_Zc'])},"

    # viscAh: %g rescues small values (e.g. Gong's 0.01 floor) that the old
    # %.1f silently truncated to 0.0, while keeping round magnitudes (500.0,
    # 0.0) byte-identical so existing cases' namelists do not change.
    viscah_text = f"{viscosity:.6g}"
    if all(c not in viscah_text for c in (".", "e", "E")):
        viscah_text = f"{viscah_text}.0"
    viscah_text = _exact(viscosity, viscah_text)

    # Horizontal viscosity scheme: when leith_smag, comment scalar viscAh and
    # emit Leith+Smagorinsky coefficients; viscAh stays as the floor in MITgcm
    # MOM_CALC_VISC (`pkg/mom_common/mom_calc_visc.F:503,650`).
    if viscosity_scheme in ("leith_smag", "leith_smag_flux"):
        visc_lines = (
            f"# viscAh={viscah_text},  # commented — Leith+Smag variable viscosity\n"
            f" viscAh={viscah_text},\n"
            f" viscA4={float(viscA4)!r},\n"
            f" viscC2Leith={visc_c2_leith},\n"
            f" viscC2smag={visc_c2_smag},"
        )
    else:
        visc_lines = (
            f" viscAh={viscah_text},\n"
            f" viscA4={float(viscA4)!r},\n"
            f" viscC2Leith=0.,\n"
            f" viscC2smag=0.,"
        )
    if viscosity_extra_lines:
        visc_lines = visc_lines + "\n" + viscosity_extra_lines.rstrip("\n")

    # Grid-specific lines
    coriolis_lines = ""
    if use_cartesian:
        assert None not in (dx_m, dy_m, f0, beta), "dx_m, dy_m, f0 and beta required for Cartesian grid"
        coriolis_lines = f""" f0={_namelist_real(f0)},
 beta={_namelist_real(beta)},"""
        grid_lines = f""" usingCartesianGrid=.TRUE.,
 delX={nx}*{dx_m:.6f},
 delY={ny}*{dy_m:.6f},
{delz_line}"""
    else:
        grid_lines = f""" usingSphericalPolarGrid=.TRUE.,
 delX={nx}*{_exact(resolution_deg, f"{resolution_deg:.10f}")},
 delY={ny}*{_exact(resolution_deg, f"{resolution_deg:.10f}")},
{delz_line}
 ygOrigin={_exact(yg_origin, f"{yg_origin:.10f}")},
 xgOrigin={_exact(xg_origin, f"{xg_origin:.10f}")},"""

    implicit_diff_str = ".TRUE." if implicit_diffusion else ".FALSE."
    implicit_visc_str = ".TRUE." if implicit_viscosity else ".FALSE."
    temp_impl_vert_adv_str = (
        ".TRUE." if temperature_implicit_vertical_advection else ".FALSE."
    )
    salt_impl_vert_adv_str = (
        ".TRUE." if salinity_implicit_vertical_advection else ".FALSE."
    )
    mom_impl_vert_adv_str = (
        ".TRUE." if momentum_implicit_vertical_advection else ".FALSE."
    )
    tracer_kappa_r_text = _namelist_real(tracer_vertical_diffusivity_kr)
    diffk4_lines = ""
    if diffusivity_k4_temp != 0.0:
        diffk4_lines += f" diffK4T={_namelist_real(diffusivity_k4_temp)},\n"
    if diffusivity_k4_salt != 0.0:
        diffk4_lines += f" diffK4S={_namelist_real(diffusivity_k4_salt)},\n"
    # viscAz: emit the resolved vertical viscosity (== TO stepper
    # implicit_viscosity).
    viscaz_text = _namelist_real(vertical_viscosity)
    no_slip_sides_str = ".TRUE." if no_slip_sides else ".FALSE."
    no_slip_bottom_str = ".TRUE." if no_slip_bottom else ".FALSE."
    exact_conserv_str = ".TRUE." if exact_conserv else ".FALSE."
    stagger_time_step_str = ".TRUE." if stagger_time_step else ".FALSE."
    temp_stepping_str = ".TRUE." if temp_stepping else ".FALSE."
    salt_stepping_str = ".TRUE." if salt_stepping else ".FALSE."

    def _real_text(value: float) -> str:
        text = f"{float(value):.15G}".replace("e", "E")
        if "E" not in text and "." not in text:
            text = f"{text}.0"
        return text

    bottom_drag_linear_text = _exact(bottom_drag_linear, _real_text(bottom_drag_linear))
    bottom_drag_quadratic_text = _exact(bottom_drag_quadratic, _real_text(bottom_drag_quadratic))
    # repr: the shortest text that reads back as the same double, which the
    # consumer's re-parse compares exactly (0.1 still renders as "0.1").
    ab_eps_text = repr(float(ab_eps)).replace("e", "E")
    lin_fs_conserve_tr_str = ".TRUE." if lin_fs_conserve_tr else ".FALSE."
    nonhydrostatic_str = ".TRUE." if nonhydrostatic else ".FALSE."
    cg3d_target_residual_text = _exact(cg3d_target_residual, _e_text(cg3d_target_residual))
    cg3d_target_res_w_unit_text = _exact(cg3d_target_res_w_unit, _e_text(cg3d_target_res_w_unit))
    nh_am2_text = _exact(nh_am2, _e_text(nh_am2))
    nh_lines = ""
    cg3d_target_res_w_unit_line = ""
    # implicSurfPress / implicDiv2DFlow: MITgcm defaults both to 1
    # (set_defaults.F:252-253), LOCOMgcm's stepper to 0.5; written when they
    # differ from 1, and always under the nonhydrostatic block.
    implicit_lines = ""
    if nonhydrostatic or float(implic_surf_press) != 1.0 or float(implic_div_2d_flow) != 1.0:
        implicit_lines = (f" implicSurfPress={_exact(implic_surf_press, _e_text(implic_surf_press))},\n"
                          f" implicDiv2DFlow={_exact(implic_div_2d_flow, _e_text(implic_div_2d_flow))},\n")
    if nonhydrostatic:
        nh_lines = f""" implicitNHPress={_exact(implic_surf_press, _e_text(implic_surf_press))},
 selectNHfreeSurf=0,
 nh_Am2={nh_am2_text},
"""
        cg3d_target_res_w_unit_line = (
            f" cg3dTargetResWunit={cg3d_target_res_w_unit_text},\n"
        )

    adv_scheme_lines = "".join(
        f" {key}={value},\n"
        for key, value in (("tempAdvScheme", tempAdvScheme), ("saltAdvScheme", saltAdvScheme))
        if value is not None
    )
    cg2d_lines = f" cg2dMaxIters={int(cg2d['cg2dMaxIters'])},\n"
    for key in ("cg2dTargetResidual", "cg2dTargetResWunit"):
        if key in cg2d:
            cg2d_lines += f" {key}={_exact(cg2d[key], _e_text(cg2d[key]))},\n"
    momentum_form = str(momentum_advection_form)
    if momentum_form not in {"vector_invariant", "flux_form"}:
        raise ValueError(
            "momentum_advection_form must be 'vector_invariant' or 'flux_form', "
            f"got {momentum_form!r}"
        )
    vec_inv_str = ".TRUE." if momentum_form == "vector_invariant" else ".FALSE."

    text = f"""\
# MITgcm data — generated by mitgcm_export.py
# Aligned with LOCOMgcm compare_hycom.py parameters
#
 &PARM01
{tref_line}
{sref_line}
 no_slip_sides={no_slip_sides_str},
 no_slip_bottom={no_slip_bottom_str},
 bottomDragLinear={bottom_drag_linear_text},
 bottomDragQuadratic={bottom_drag_quadratic_text},
 selectKEscheme={ke_scheme},
{visc_lines}
 viscAz={viscaz_text},
 diffKhT={_namelist_real(diffusivity_t)},
 diffKzT={tracer_kappa_r_text},
 diffKhS={_namelist_real(diffusivity_s)},
 diffKzS={tracer_kappa_r_text},
{diffk4_lines} rigidLid=.FALSE.,
 implicitFreeSurface=.TRUE.,
 exactConserv={exact_conserv_str},
 linFSConserveTr={lin_fs_conserve_tr_str},
 nonHydrostatic={nonhydrostatic_str},
{implicit_lines}{nh_lines}{eos_lines}
{coriolis_lines}
 rhoConst={_namelist_real(rho_const)},
{rho_nil_line} gravity={_namelist_real(gravity)},
 saltStepping={salt_stepping_str},
 tempStepping={temp_stepping_str},
 momStepping={".TRUE." if mom_stepping else ".FALSE."},
 tempImplVertAdv={temp_impl_vert_adv_str},
 saltImplVertAdv={salt_impl_vert_adv_str},
 momImplVertAdv={mom_impl_vert_adv_str},
 vectorInvariantMomentum={vec_inv_str},
 implicitDiffusion={implicit_diff_str},
 implicitViscosity={implicit_visc_str},
 staggerTimeStep={stagger_time_step_str},
{adv_scheme_lines}{_format_vert_adv_scheme(tempVertAdvScheme, saltVertAdvScheme)} multiDimAdvection={".TRUE." if multi_dim_advection else ".FALSE."},
 readBinaryPrec=64,
 writeBinaryPrec=64,
 useSingleCpuIO=.TRUE.,
 hFacMin={_exact(hfac_min, f"{hfac_min:.1f}")},
 hFacMinDz={_exact(hfac_min_dz, f"{hfac_min_dz:.15g}")},
 &

# Elliptic solver
 &PARM02
{cg2d_lines} cg3dMaxIters={cg3d_max_iters},
 cg3dTargetResidual={cg3d_target_residual_text},
{cg3d_target_res_w_unit_line} &

# Time stepping
 &PARM03
 startTime=0.0,
 endTime={_exact(total_seconds, f"{total_seconds:.1f}")},
 deltaT={_exact(dt_seconds, f"{dt_seconds:.15g}")},
 deltaTmom={_exact(dt_mom, f"{dt_mom:.15g}")},
 deltaTtracer={_exact(dt_tracer, f"{dt_tracer:.15g}")},
 deltaTFreeSurf={_exact(dt_freesurf, f"{dt_freesurf:.15g}")},
 deltaTClock={_exact(dt_clock, f"{dt_clock:.15g}")},
 cAdjFreq={_exact(c_adj_freq, "0.")},
 abEps={ab_eps_text},
 momForcingOutAB={mom_forcing_out_ab},
 tracForcingOutAB={trac_forcing_out_ab},
 pChkptFreq=0.,
 chkptFreq=0.,
 dumpFreq={_exact(dump_freq, f"{dump_freq:.1f}")},
 monitorFreq={_exact(monitor_freq, f"{monitor_freq:.1f}")},
 monitorSelect=2,
 pickupStrictlyMatch=.FALSE.,
 &

# Grid
 &PARM04
{grid_lines}
 &

# Input files
 &PARM05
 bathyFile='bathyFile.bin',
 hydrogThetaFile='hydrogThetaFile.bin',
 hydrogSaltFile='hydrogSaltFile.bin',
 uVelInitFile='uVelInitFile.bin',
 vVelInitFile='vVelInitFile.bin',
 pSurfInitFile='pSurfInitFile.bin',
 checkIniTemp=.FALSE.,
 checkIniSalt=.FALSE.,
 &
"""
    # MITgcm would silently drop everything past column MITGCM_MAX_LEN_PREC.
    long_lines = [
        f"line {n}: {len(line)} characters, starts {line[:24]!r}"
        for n, line in enumerate(text.splitlines(), start=1)
        if len(line) > MITGCM_MAX_LEN_PREC
    ]
    if long_lines:
        raise ValueError(
            f"{path}: MITgcm reads only the first {MITGCM_MAX_LEN_PREC} characters "
            "of a namelist line; wrap these lines:\n  " + "\n  ".join(long_lines)
        )
    path.write_text(text)


def _write_data_pkg(
    path: Path,
    *,
    use_exf: bool = True,
    use_kpp: bool = False,
    use_diagnostics: bool = False,
) -> None:
    """Generate data.pkg.

    ``use_exf`` defaults True so existing cases retain their EXF setting.
    No-wind cases pass False (Gong has no zonalWindFile) so MITgcm does not
    require data.exf / wind binaries.

    useCAL FOLLOWS use_exf: pkg/cal is only compiled as an EXF dependency (it is
    NOT listed in packages.conf on its own), so dropping exf also drops cal from
    the build. Keeping useCAL=.TRUE. without exf makes PACKAGES_CHECK abort
    ("useCAL is set but pkg/cal was not compiled"). Existing exf cases keep
    useCAL=.TRUE. (cal present via exf); no-wind cases get useCAL=.FALSE.
    """
    exf_line = " useEXF=.TRUE.,\n" if use_exf else " useEXF=.FALSE.,\n"
    cal_line = " useCAL=.TRUE.,\n" if use_exf else " useCAL=.FALSE.,\n"
    kpp_line = " useKPP=.TRUE.,\n" if use_kpp else " useKPP=.FALSE.,\n"
    diagnostics_line = (
        " useDiagnostics=.TRUE.,\n"
        if use_diagnostics
        else " useDiagnostics=.FALSE.,\n"
    )
    path.write_text(
        "# Packages — generated by mitgcm_export.py\n"
        " &PACKAGES\n"
        f"{exf_line}"
        f"{cal_line}"
        " useOBCS=.TRUE.,\n"
        f"{kpp_line}"
        f"{diagnostics_line}"
        " &\n"
    )


def _write_data_diagnostics(path: Path) -> None:
    """Generate the six-hour mean UVELMASS diagnostics stream."""
    path.write_text("""\
 &DIAGNOSTICS_LIST
  diag_mnc = .FALSE.,
  fields(1,1) = 'UVELMASS',
  fileName(1) = 'UVELMASS_6h',
  frequency(1) = 21600.,
  timePhase(1) = 0.,
 &
 &DIAG_STATIS_PARMS
 &
""")


def _write_data_kpp(path: Path) -> None:
    """Generate the minimal KPP namelist used by benchmark exports.

    ``KPPwriteState`` stays ``.FALSE.``: these exports feed steady-state
    per-step timing runs, and the diagnostic dump it enables is exactly the
    in-window I/O such a measurement must not carry.
    """
    path.write_text("""\
# KPP parameters -- generated by mitgcm_export.py
 &KPP_PARM01
 KPPmixingMaps=.FALSE.,
 KPPwriteState=.FALSE.,
 KPP_ghatUseTotalDiffus=.TRUE.,
 &
""")


def _write_data_obcs(
    path: Path,
    *,
    nx: int,
    ny: int,
    sponge_width: int,
    sponge_timescale_seconds: float,
    sponge_timescale_inner_seconds: float,
    use_linear_sponge: bool,
    obcs_period: float,
    sponge_relax_tracers: bool = False,
) -> None:
    """Generate data.obcs with Orlanski + sponge + prescribe + file names.

    Parameters
    ----------
    sponge_relax_tracers : bool
        If True, also emit Trelaxobcsinner/bound and Srelaxobcsinner/bound
        so MITgcm relaxes T/S in the sponge layer (matching LOCOMgcm's
        TracerBoundarySpongeRelaxation). Default False preserves the
        historical U/V-only sponge.
    """
    # Relaxation timescales in days for MITgcm sponge
    # MITgcm: Urelaxobcsbound = timescale at boundary (days)
    #         Urelaxobcsinner = timescale at inner edge (days)
    bound_days = sponge_timescale_seconds / 86400.0
    inner_days = sponge_timescale_inner_seconds / 86400.0

    use_linear_sponge_line = " useLinearSponge=.TRUE.,\n" if use_linear_sponge else ""

    # NOTE: MITgcm's OBCS_SPONGE_T/S reuses Vrelaxobcsbound/inner for tracer
    # sponge timescales (OBCSsponge_Theta=.TRUE. by default).  No separate
    # Trelaxobcs*/Srelaxobcs* namelist parameters exist.

    path.write_text(f"""\
# OBCS — generated by mitgcm_export.py
# Prescribed boundary condition for U/V/T/S with Sponge layer
 &OBCS_PARM01
 OB_Jsouth = {nx}*1,
 OB_Jnorth = {nx}*{ny},
 OB_Iwest  = {ny}*1,
 OB_Ieast  = {ny}*{nx},
 OBCSfixTopo=.TRUE.,
 useOBCSprescribe=.TRUE.,
 useOBCSbalance=.TRUE.,
 useOBCSsponge=.TRUE.,
 OBCS_monitorFreq={obcs_period},
 OBNtFile='OBNt.bin',
 OBNsFile='OBNs.bin',
 OBStFile='OBSt.bin',
 OBSsFile='OBSs.bin',
 OBEtFile='OBEt.bin',
 OBEsFile='OBEs.bin',
 OBWtFile='OBWt.bin',
 OBWsFile='OBWs.bin',
 OBNvFile='OBNv.bin',
 OBSvFile='OBSv.bin',
 OBEuFile='OBEu.bin',
 OBWuFile='OBWu.bin',
 OBNuFile='OBNu.bin',
 OBSuFile='OBSu.bin',
 OBEvFile='OBEv.bin',
 OBWvFile='OBWv.bin',
 &

 &OBCS_PARM02
 &

 &OBCS_PARM03
{use_linear_sponge_line} spongeThickness = {sponge_width},
 Urelaxobcsinner = {sponge_timescale_inner_seconds:.1f},
 Vrelaxobcsinner = {sponge_timescale_inner_seconds:.1f},
 Urelaxobcsbound = {sponge_timescale_seconds:.1f},
 Vrelaxobcsbound = {sponge_timescale_seconds:.1f},
 &
""")


def _write_data_exf(
    path: Path,
    *,
    exf_period: float,
    obcs_period: float,
    windstressmax: float,
    obcs_repeat_period: float = 0.0,
    start_date_ymd: int = MITGCM_START_DATE_YMD,
) -> None:
    """Generate data.exf for direct wind stress input.

    ``windstressmax`` is the clip LOCOMgcm's ``era5_wind_forcing`` applies to
    the centre stress (``layer_d/_mit_switches.windstressmax_value``); it has no
    default here so no writer can fall back to a constant of its own.
    """
    path.write_text(f"""\
# EXF — generated by mitgcm_export.py
# Direct wind stress input (tau_x, tau_y already computed)
 &EXF_NML_01
 useExfCheckRange=.FALSE.,
 exf_iprec=64,
 useExfYearlyFields=.FALSE.,
 repeatPeriod={obcs_repeat_period:.1f},
 useAtmWind=.FALSE.,
 readStressOnAgrid=.TRUE.,
 windstressmax={render_windstressmax(windstressmax)},
 &

 &EXF_NML_02
 ustressperiod={exf_period:.1f},
 vstressperiod={exf_period:.1f},
 ustressstartdate1={start_date_ymd},
 ustressstartdate2=000000,
 vstressstartdate1={start_date_ymd},
 vstressstartdate2=000000,
 ustressfile='ustress.bin',
 vstressfile='vstress.bin',
 &

 &EXF_NML_03
 &

 &EXF_NML_OBCS
 obcsSperiod={obcs_period:.1f},
 obcsNperiod={obcs_period:.1f},
 obcsWperiod={obcs_period:.1f},
 obcsEperiod={obcs_period:.1f},
 obcsSstartdate1={start_date_ymd},
 obcsSstartdate2=000000,
 obcsNstartdate1={start_date_ymd},
 obcsNstartdate2=000000,
 obcsWstartdate1={start_date_ymd},
 obcsWstartdate2=000000,
 obcsEstartdate1={start_date_ymd},
 obcsEstartdate2=000000,
 &
""")


def _write_data_cal(path: Path, *, start_date_ymd: int = 20190101) -> None:
    """Generate data.cal (calendar configuration)."""
    path.write_text(f"""\
# CAL — generated by mitgcm_export.py
 &CAL_NML
 TheCalendar='gregorian',
 startDate_1={start_date_ymd},
 startDate_2=000000,
 &
""")


def _write_eedata(path: Path) -> None:
    """Generate eedata (execution environment)."""
    path.write_text("""\
# Execution Environment — generated by mitgcm_export.py
 &EEPARMS
 useCubedSphereExchange=.FALSE.,
 &
 &EESUPPORT
 nThreads=1,
 &
""")


# ──────────────────────────────────────────────────────────────────────
# Binary data export
# ──────────────────────────────────────────────────────────────────────

def _export_initial_conditions(
    input_dir: Path,
    *,
    init_temp: np.ndarray,
    init_salt: np.ndarray,
    init_u: np.ndarray,
    init_v: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Export IC fields as big-endian float64 binary."""
    projected = {
        "temp": np.asarray(init_temp, dtype=np.float64),
        "salt": np.asarray(init_salt, dtype=np.float64),
        "u": np.asarray(init_u, dtype=np.float64),
        "v": np.asarray(init_v, dtype=np.float64),
    }
    _write_binary(input_dir / "hydrogThetaFile.bin", projected["temp"])
    _write_binary(input_dir / "hydrogSaltFile.bin", projected["salt"])
    _write_binary(input_dir / "uVelInitFile.bin", projected["u"])
    _write_binary(input_dir / "vVelInitFile.bin", projected["v"])


def _export_bathymetry(
    input_dir: Path,
    *,
    bathymetry: np.ndarray | torch.Tensor,
) -> None:
    """Export bathymetry as big-endian float64."""
    _write_binary(input_dir / "bathyFile.bin", project_bathymetry_to_mitgcm(bathymetry))


def _export_obcs_files(
    input_dir: Path,
    *,
    boundary_source: Any,  # HYCOMOpenBoundarySource
) -> None:
    """Export OBCS binary files."""
    for filename, arr in project_boundary_source_to_mitgcm_files(boundary_source).items():
        _write_binary(input_dir / filename, arr)


def _export_exf_files(
    input_dir: Path,
    *,
    era5_source: Any,  # ERA5WindSource
) -> None:
    """Export EXF wind stress binary files."""
    projected = project_era5_source_to_mitgcm_files(era5_source)
    _write_binary(input_dir / "ustress.bin", projected["ustress.bin"])
    _write_binary(input_dir / "vstress.bin", projected["vstress.bin"])


# ──────────────────────────────────────────────────────────────────────
# Main export function
# ──────────────────────────────────────────────────────────────────────

def export_mitgcm_case(
    output_dir: Path,
    *,
    mitgcm_root: Path | None = None,
    # Grid
    nx: int,
    ny: int,
    nz: int,
    resolution_deg: float,
    center_lat: float,
    center_lon: float,
    depth_m: float,
    dt_seconds: float,
    total_seconds: float,
    layer_thickness: Any | None = None,
    dt_mom_seconds: float | None = None,
    dt_tracer_seconds: float | None = None,
    dt_freesurf_seconds: float | None = None,
    dt_clock_seconds: float | None = None,
    # Physics
    viscosity: float,
    viscA4: float,
    diffusivity: float,
    # Sponge
    sponge_width: int = 0,
    sponge_timescale_seconds: float = 3600.0,
    sponge_timescale_inner_seconds: float = 36000.0,
    use_linear_sponge: bool = False,
    sponge_relax_tracers: bool = False,
    # Output
    dump_freq_seconds: float = 10800.0,
    # Data
    bathymetry: Any,
    init_temp: Any,
    init_salt: Any,
    init_u: Any,
    init_v: Any,
    boundary_source: Any,
    era5_source: Any,
    temperature_implicit_vertical_advection: bool,
    salinity_implicit_vertical_advection: bool,
    momentum_implicit_vertical_advection: bool,
    tempAdvScheme: int = 77,
    saltAdvScheme: int = 77,
    tempVertAdvScheme: int | None = None,
    saltVertAdvScheme: int | None = None,
    # PARM01 equation of state and constants of the LOCOMgcm run (see _write_data)
    eos: dict[str, Any],
    rho_const: float,
    gravity: float,
    # Ablation experiment controls
    use_cartesian: bool = False,
    f0: float | None = None,
    beta: float | None = None,
    implicit_diffusion: bool = DEFAULT_IMPLICIT_DIFFUSION,
    tracer_vertical_diffusivity_kr: float = DEFAULT_DIFFUSIVITY_KR,
    implicit_viscosity: bool = True,
    # viscAz: the vertical viscosity the caller's LOCOMgcm model runs.
    vertical_viscosity: float,
    force_dt: bool = False,
    hfac_min: float = MITGCM_HFAC_MIN,
    hfac_min_dz: float = MITGCM_HFAC_MIN_DZ,
    lin_fs_conserve_tr: bool = False,
    ke_scheme: int = 0,
    viscosity_scheme: str = "constant",
    visc_c2_leith: float = 0.0,
    visc_c2_smag: float = 0.0,
    momentum_advection_form: str = "vector_invariant",
    enable_vertical_shear: bool = True,
    nonhydrostatic: bool | NonhydrostaticParams = False,
    cg3d_max_iters: int | None = None,
    cg3d_target_residual: float | None = None,
    windstressmax: float = WINDSTRESSMAX_EXF_DEFAULT,
) -> Path:
    """Create a complete MITgcm case directory from LOCOMgcm data.

    Parameters
    ----------
    output_dir : Path
        Root directory for the MITgcm case (code/, input/, build/, run/).
    nx, ny, nz : int
        Grid dimensions.
    resolution_deg : float
        Regular grid spacing in degrees.
    center_lat, center_lon : float
        Patch center coordinates.
    depth_m : float
        Total ocean depth in meters.
    dt_seconds : float
        Time step in seconds.
    total_seconds : float
        Total integration time in seconds.
    viscosity, diffusivity : float
        Horizontal viscosity and tracer diffusivity [m²/s].
    viscA4 : float
        Horizontal biharmonic viscosity [m^4/s].
    sponge_width : int
        Sponge layer thickness in grid cells.
    sponge_timescale_seconds : float
        Sponge relaxation at boundary edge [s].
    sponge_timescale_inner_seconds : float
        Sponge relaxation at inner edge [s].
    use_linear_sponge : bool
        Export MITgcm ``useLinearSponge=.TRUE.`` when requested.
    sponge_relax_tracers : bool
        If True, emit Trelaxobcsinner/bound and Srelaxobcsinner/bound
        so MITgcm relaxes T/S in the sponge layer (matching LOCOMgcm's
        TracerBoundarySpongeRelaxation). Default False = U/V-only sponge.
    dump_freq_seconds : float
        MITgcm output dump frequency [s].
    bathymetry : Tensor/ndarray [ny, nx]
        Bathymetry, positive = ocean depth.
    init_temp, init_salt : Tensor/ndarray [nz, ny, nx]
        Initial temperature and salinity.
    init_u : Tensor/ndarray [nz, ny, nx+1]
        Initial zonal velocity (C-grid staggered).
    init_v : Tensor/ndarray [nz, ny+1, nx]
        Initial meridional velocity (C-grid staggered).
    boundary_source : HYCOMOpenBoundarySource
        Time-varying open boundary fields.
    era5_source : ERA5WindSource
        Time-varying wind stress fields.
    nonhydrostatic : bool or NonhydrostaticParams
        Enable MITgcm nonhydrostatic/CG3D build and namelist controls.
    cg3d_max_iters, cg3d_target_residual : optional
        Override CG3D solver controls.  If ``nonhydrostatic`` is a
        NonhydrostaticParams object and these are omitted, its values are used.
    windstressmax : float
        The ``windstressmax`` of the ``ERA5WindForcing`` the LOCOMgcm run
        uses (its default when omitted); ``inf`` means that module runs
        without the clip.

    Returns
    -------
    Path : the case root directory (output_dir).
    """
    temperature_implicit_vertical_advection = _require_exact_bool(
        temperature_implicit_vertical_advection,
        name="temperature_implicit_vertical_advection",
    )
    salinity_implicit_vertical_advection = _require_exact_bool(
        salinity_implicit_vertical_advection,
        name="salinity_implicit_vertical_advection",
    )
    momentum_implicit_vertical_advection = _require_exact_bool(
        momentum_implicit_vertical_advection,
        name="momentum_implicit_vertical_advection",
    )
    # Create directory structure
    code_dir = output_dir / "code"
    input_dir = output_dir / "input"
    build_dir = output_dir / "build"
    run_dir = output_dir / "run"
    for d in [code_dir, input_dir, build_dir, run_dir]:
        d.mkdir(parents=True, exist_ok=True)

    (
        nonhydrostatic_enabled,
        resolved_cg3d_max_iters,
        resolved_cg3d_target_residual,
        resolved_cg3d_target_res_w_unit,
        resolved_nh_am2,
    ) = _resolve_nonhydrostatic_export_controls(
        nonhydrostatic,
        cg3d_max_iters,
        cg3d_target_residual,
    )

    # Convert tensors to numpy
    init_temp_np = np.asarray(_to_numpy(init_temp), dtype=np.float64)
    init_salt_np = np.asarray(_to_numpy(init_salt), dtype=np.float64)
    if init_temp_np.ndim == 4:
        init_temp_np = init_temp_np[0]
    if init_salt_np.ndim == 4:
        init_salt_np = init_salt_np[0]

    projected_initial = project_state_to_mitgcm_initial_fields(
        type("_ExportState", (), {
            "eta": type("_Field", (), {"data": torch.zeros((1, ny, nx), dtype=torch.float64)})(),
            "temp": type("_Field", (), {"data": init_temp})(),
            "salt": type("_Field", (), {"data": init_salt})(),
            "u": type("_Field", (), {"data": init_u})(),
            "v": type("_Field", (), {"data": init_v})(),
        })(),
        nx=nx,
        ny=ny,
        nz=nz,
    )

    # Grid origin: convert from center to southwest corner
    xg_origin = center_lon - (nx / 2) * resolution_deg
    yg_origin = center_lat - (ny / 2) * resolution_deg

    # OBCS period: time between boundary records
    boundary_times = boundary_source.times
    if isinstance(boundary_times, torch.Tensor):
        boundary_times = boundary_times.detach().cpu().numpy()
    else:
        boundary_times = np.asarray(boundary_times)
    if len(boundary_times) > 1:
        obcs_period = float(boundary_times[1] - boundary_times[0])
        # total span of boundary data for cyclic repeat
        obcs_repeat_period = float(boundary_times[-1])
    else:
        obcs_period = total_seconds
        obcs_repeat_period = total_seconds

    # EXF period: time between ERA5 records
    era5_times = era5_source.times
    if isinstance(era5_times, torch.Tensor):
        era5_times = era5_times.detach().cpu().numpy()
    else:
        era5_times = np.asarray(era5_times)
    if len(era5_times) > 1:
        exf_period = float(era5_times[1] - era5_times[0])
    else:
        exf_period = total_seconds

    # Compute CFL-safe timestep for MITgcm
    # MITgcm uses explicit advection (unlike LOCOMgcm's split-explicit stepper),
    # so it needs a much smaller dt.
    import math
    dx_m = resolution_deg * 111_320 * math.cos(math.radians(center_lat))
    dy_m = resolution_deg * 110_574
    dz_m = depth_m / nz
    init_u_np = np.asarray(_to_numpy(init_u), dtype=np.float64)
    init_v_np = np.asarray(_to_numpy(init_v), dtype=np.float64)
    max_u = float(np.abs(init_u_np).max())
    max_v = float(np.abs(init_v_np).max())
    max_speed = max(max_u, max_v, 0.1)
    if force_dt:
        mitgcm_dt = dt_seconds
    else:
        # CFL: dt <= dx / u_max * safety_factor (0.3 for AB2)
        cfl_dt_h = min(dx_m, dy_m) / max_speed * 0.3
        # Vertical CFL is usually fine with implicit diffusion
        mitgcm_dt = min(dt_seconds, cfl_dt_h, 120.0)  # cap at 120s max
        mitgcm_dt = max(mitgcm_dt, 1.0)  # floor at 1s

    # Adjust total_seconds to be integer multiple of mitgcm_dt.
    # IMPORTANT: MITgcm cal_Set requires deltaTClock to be an integer number
    # of seconds.  Floor the computed dt and recompute n_steps.
    mitgcm_dt = max(1.0, math.floor(mitgcm_dt))
    mitgcm_dt_mom = mitgcm_dt if dt_mom_seconds is None else dt_mom_seconds
    mitgcm_dt_tracer = mitgcm_dt if dt_tracer_seconds is None else dt_tracer_seconds
    mitgcm_dt_freesurf = mitgcm_dt if dt_freesurf_seconds is None else dt_freesurf_seconds
    mitgcm_dt_clock = mitgcm_dt if dt_clock_seconds is None else dt_clock_seconds
    # Route the step count through the single-source helper so
    # this legacy export path can never disagree with the LOCOMgcm runtime on n_steps.
    # ``total_seconds`` here is a positional float (no `days` channel), so
    # we feed ``days = total_seconds / 86400`` to compute_n_steps; the same
    # ULP-trap detection applies. The helper returns ``round``-based count
    # and we recompute ``adjusted_total = n_steps * dt`` from it.
    from defgcm.io.runner._time_steps import compute_n_steps, exact_total_seconds, SECONDS_PER_DAY
    days_equivalent = float(total_seconds) / SECONDS_PER_DAY
    n_steps = compute_n_steps(
        days=days_equivalent,
        dt_seconds=mitgcm_dt_clock,
        steps_override=None,
        context="experiments/alignment/mitgcm_export.export_mitgcm_case",
    )
    adjusted_total = exact_total_seconds(n_steps, mitgcm_dt_clock)

    print(f"  [MITgcm export] Grid: {nx}×{ny}×{nz}, "
          f"res={resolution_deg:.6f}°, depth={depth_m:.1f}m", flush=True)
    print(f"  [MITgcm export] LOCOMgcm dt={dt_seconds:.2f}s, "
          f"MITgcm dt={mitgcm_dt:.2f}s, dtClock={mitgcm_dt_clock:.2f}s "
          f"({n_steps} steps)", flush=True)
    print(f"  [MITgcm export] OBCS period={obcs_period:.0f}s, "
          f"EXF period={exf_period:.0f}s", flush=True)

    # ── Code files ──
    _write_size_h(code_dir / "SIZE.h", nx=nx, ny=ny, nz=nz)
    _write_packages_conf(code_dir / "packages.conf")
    _write_cpp_options(
        code_dir / "CPP_OPTIONS.h",
        nonhydrostatic=nonhydrostatic_enabled,
    )
    _write_obcs_options(code_dir / "OBCS_OPTIONS.h", sponge_width=sponge_width)
    _write_exf_options(code_dir / "EXF_OPTIONS.h")
    if momentum_advection_form == "vector_invariant" and not enable_vertical_shear:
        _patch_disable_vertshear(code_dir, mitgcm_root)

    # ── Configuration files ──
    _write_data(
        input_dir / "data",
        nx=nx, ny=ny, nz=nz,
        resolution_deg=resolution_deg,
        depth_m=depth_m,
        layer_thickness=layer_thickness,
        dt_seconds=mitgcm_dt,
        dt_mom_seconds=mitgcm_dt_mom,
        dt_tracer_seconds=mitgcm_dt_tracer,
        dt_freesurf_seconds=mitgcm_dt_freesurf,
        dt_clock_seconds=mitgcm_dt_clock,
        total_seconds=adjusted_total,
        viscosity=viscosity,
        viscA4=viscA4,
        diffusivity_t=diffusivity,
        diffusivity_s=diffusivity,
        xg_origin=xg_origin,
        yg_origin=yg_origin,
        dump_freq=dump_freq_seconds,
        init_temp=init_temp_np,
        init_salt=init_salt_np,
        temperature_implicit_vertical_advection=(
            temperature_implicit_vertical_advection
        ),
        salinity_implicit_vertical_advection=salinity_implicit_vertical_advection,
        momentum_implicit_vertical_advection=momentum_implicit_vertical_advection,
        tempAdvScheme=tempAdvScheme,
        saltAdvScheme=saltAdvScheme,
        tempVertAdvScheme=tempVertAdvScheme,
        saltVertAdvScheme=saltVertAdvScheme,
        eos=eos,
        rho_const=rho_const,
        gravity=gravity,
        use_cartesian=use_cartesian,
        dx_m=dx_m,
        dy_m=dy_m,
        f0=f0,
        beta=beta,
        implicit_diffusion=implicit_diffusion,
        tracer_vertical_diffusivity_kr=tracer_vertical_diffusivity_kr,
        implicit_viscosity=implicit_viscosity,
        vertical_viscosity=vertical_viscosity,
        hfac_min=hfac_min,
        hfac_min_dz=hfac_min_dz,
        lin_fs_conserve_tr=lin_fs_conserve_tr,
        ke_scheme=ke_scheme,
        viscosity_scheme=viscosity_scheme,
        visc_c2_leith=visc_c2_leith,
        visc_c2_smag=visc_c2_smag,
        momentum_advection_form=momentum_advection_form,
        nonhydrostatic=nonhydrostatic_enabled,
        cg3d_max_iters=resolved_cg3d_max_iters,
        cg3d_target_residual=resolved_cg3d_target_residual,
        cg3d_target_res_w_unit=resolved_cg3d_target_res_w_unit,
        nh_am2=resolved_nh_am2,
    )
    _write_data_pkg(input_dir / "data.pkg")
    _write_data_obcs(
        input_dir / "data.obcs",
        nx=nx, ny=ny,
        sponge_width=sponge_width,
        sponge_timescale_seconds=sponge_timescale_seconds,
        sponge_timescale_inner_seconds=sponge_timescale_inner_seconds,
        use_linear_sponge=use_linear_sponge,
        obcs_period=obcs_period,
        sponge_relax_tracers=sponge_relax_tracers,
    )
    _write_data_exf(
        input_dir / "data.exf",
        exf_period=exf_period,
        obcs_period=obcs_period,
        windstressmax=windstressmax_namelist_value(windstressmax),
        obcs_repeat_period=obcs_repeat_period,
    )
    _write_data_cal(input_dir / "data.cal")
    _write_eedata(input_dir / "eedata")

    # ── Binary data files ──
    print("  [MITgcm export] Writing initial conditions ...", flush=True)
    _export_initial_conditions(
        input_dir,
        init_temp=projected_initial["temp"],
        init_salt=projected_initial["salt"],
        init_u=projected_initial["u"],
        init_v=projected_initial["v"],
        nx=nx, ny=ny, nz=nz,
    )

    print("  [MITgcm export] Writing bathymetry ...", flush=True)
    _export_bathymetry(input_dir, bathymetry=bathymetry)

    print("  [MITgcm export] Writing OBCS files ...", flush=True)
    _export_obcs_files(input_dir, boundary_source=boundary_source)

    print("  [MITgcm export] Writing EXF wind stress ...", flush=True)
    _export_exf_files(input_dir, era5_source=era5_source)

    print(f"  [MITgcm export] Case directory ready: {output_dir}", flush=True)
    return output_dir
