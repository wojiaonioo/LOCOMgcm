"""Offline G2 Tier-0 geometry producer.

The returned mapping is the ``tier0`` node of a ``to-g2-history/v1``
artifact.  Grid statistics follow MITgcm ``MON_STATS_RS``: the physical
horizontal domain only, unweighted population statistics, and a two-pass
standard deviation.  ``ke_vol`` follows ``MON_KE`` and includes every
``maskInC``/``hFacC``/vertical factor explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import torch

from experiments.mitcases.g2_compare import (
    GRID_BASE_FIELDS,
    GRID_KEYS,
    GRID_STAT_SUFFIXES,
)


class _Unavailable(ValueError):
    pass


def _float64_tensor(value: object, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise _Unavailable(f"{label} is not a torch.Tensor")
    if value.dtype != torch.float64:
        raise _Unavailable(f"{label} must be torch.float64, got {value.dtype}")
    if value.numel() == 0:
        raise _Unavailable(f"{label} is empty")
    if not bool(torch.isfinite(value).all().item()):
        raise _Unavailable(f"{label} contains non-finite values")
    return value


def _normalized_fields(fields: Mapping[str, object]) -> dict[str, object]:
    by_folded_name: dict[str, object] = {}
    duplicates: set[str] = set()
    for name, value in fields.items():
        folded = str(name).casefold()
        if folded in by_folded_name:
            duplicates.add(folded)
        by_folded_name[folded] = value

    normalized: dict[str, object] = {}
    for field in GRID_BASE_FIELDS:
        folded = field.casefold()
        if folded not in duplicates and folded in by_folded_name:
            normalized[field] = by_folded_name[folded]
    return normalized


def _physical_grid_values(
    value: object,
    field: str,
    *,
    ny: int,
    nx: int,
) -> torch.Tensor:
    tensor = _float64_tensor(value, field)
    if tensor.ndim < 2 or tensor.shape[-2] < ny or tensor.shape[-1] < nx:
        raise _Unavailable(
            f"{field} cannot provide physical shape (..., {ny}, {nx}); "
            f"got {tuple(tensor.shape)}"
        )
    return tensor[..., :ny, :nx].contiguous().reshape(-1)


def _bare_stats(values: torch.Tensor) -> tuple[float, float, float, float]:
    count = values.numel()
    mean = values.sum(dtype=torch.float64) / count
    anomalies = values - mean
    variance = (anomalies * anomalies).sum(dtype=torch.float64) / count
    return (
        float(values.max().item()),
        float(values.min().item()),
        float(mean.item()),
        float(torch.sqrt(variance).item()),
    )


def _grid_statistics(
    fields: Mapping[str, object],
    *,
    ny: int,
    nx: int,
) -> dict[str, float]:
    normalized = _normalized_fields(fields)
    missing = [field for field in GRID_BASE_FIELDS if field not in normalized]
    if missing:
        raise _Unavailable(f"missing base grid fields: {missing}")

    physical = {
        field: _physical_grid_values(normalized[field], field, ny=ny, nx=nx)
        for field in GRID_BASE_FIELDS
    }
    values: dict[str, float] = {}
    for field in GRID_BASE_FIELDS:
        stats = _bare_stats(physical[field])
        values.update(
            {
                f"{field}_{suffix}": statistic
                for suffix, statistic in zip(GRID_STAT_SUFFIXES, stats, strict=True)
            }
        )
    assert set(values) == set(GRID_KEYS)
    return values


def _grid_tensor(grid: object, path: tuple[str, ...], label: str) -> torch.Tensor:
    value: Any = grid
    for component in path:
        value = getattr(value, component, None)
        if value is None:
            raise _Unavailable(f"{label} is missing")
    return _float64_tensor(value, label)


def _vertical_factor(
    value: torch.Tensor | None,
    *,
    label: str,
    nz: int,
    like: torch.Tensor,
    unit_factors_verified: bool,
) -> torch.Tensor:
    if value is None:
        if unit_factors_verified is not True:
            raise _Unavailable(
                f"{label} is missing and unit deep/density factors are not verified"
            )
        return torch.ones((nz,), dtype=torch.float64, device=like.device)
    factor = _float64_tensor(value, label)
    if tuple(factor.shape) != (nz,):
        raise _Unavailable(f"{label} must have shape ({nz},), got {tuple(factor.shape)}")
    if bool((factor <= 0).any().item()):
        raise _Unavailable(f"{label} must be positive")
    return factor.to(device=like.device)


def _ke_volume(
    grid: object,
    *,
    deep_fac2_c: torch.Tensor | None,
    rho_fac_c: torch.Tensor | None,
    unit_deep_density_factors_verified: bool,
) -> float:
    area = _grid_tensor(grid, ("metrics", "cell_area"), "rA")
    dr_f = _grid_tensor(grid, ("layer_thickness",), "drF")
    hfac_c = _grid_tensor(grid, ("masks", "hFacC"), "hFacC")
    mask_in_c = _grid_tensor(grid, ("masks", "maskInC"), "maskInC")

    if dr_f.ndim != 1:
        raise _Unavailable(f"drF must be one-dimensional, got {tuple(dr_f.shape)}")
    nz = dr_f.numel()
    expected_hfac_shape = (*area.shape[:-2], nz, *area.shape[-2:])
    if tuple(hfac_c.shape) != expected_hfac_shape:
        raise _Unavailable(
            f"hFacC must have shape {expected_hfac_shape}, got {tuple(hfac_c.shape)}"
        )
    if tuple(mask_in_c.shape) != tuple(area.shape):
        raise _Unavailable(
            f"maskInC must have shape {tuple(area.shape)}, got {tuple(mask_in_c.shape)}"
        )
    if bool(((hfac_c < 0) | (hfac_c > 1)).any().item()):
        raise _Unavailable("hFacC must be within [0, 1]")
    if bool(((mask_in_c < 0) | (mask_in_c > 1)).any().item()):
        raise _Unavailable("maskInC must be within [0, 1]")

    deep = _vertical_factor(
        deep_fac2_c,
        label="deepFac2C",
        nz=nz,
        like=area,
        unit_factors_verified=unit_deep_density_factors_verified,
    )
    rho = _vertical_factor(
        rho_fac_c,
        label="rhoFacC",
        nz=nz,
        like=area,
        unit_factors_verified=unit_deep_density_factors_verified,
    )
    factor_shape = (1,) * len(area.shape[:-2]) + (nz, 1, 1)

    # Preserve the element-wise multiplication order in mon_ke.F:77-80.
    volume = area.unsqueeze(-3)
    volume = volume * deep.reshape(factor_shape)
    volume = volume * rho.reshape(factor_shape)
    volume = volume * dr_f.to(device=area.device).reshape(factor_shape)
    volume = volume * hfac_c.to(device=area.device)
    volume = volume * mask_in_c.to(device=area.device).unsqueeze(-3)
    result = volume.sum(dtype=torch.float64)
    if not math.isfinite(float(result.item())):
        raise _Unavailable("ke_vol reduction is non-finite")
    return float(result.item())


def _produce_tier0(
    grid: object,
    fields: Mapping[str, object],
    *,
    grid_fields_semantics_verified: bool,
    deep_fac2_c: torch.Tensor | None,
    rho_fac_c: torch.Tensor | None,
    unit_deep_density_factors_verified: bool,
) -> dict[str, dict[str, float | str]]:
    values: dict[str, float] = {}
    unavailable: dict[str, str] = {}
    try:
        if grid_fields_semantics_verified is not True:
            raise _Unavailable("base grid field semantics are not verified")
        values.update(
            _grid_statistics(fields, ny=int(getattr(grid, "ny")), nx=int(getattr(grid, "nx")))
        )
    except (AttributeError, TypeError, ValueError) as exc:
        reason = str(exc)
        unavailable.update({key: reason for key in GRID_KEYS})

    try:
        values["ke_vol"] = _ke_volume(
            grid,
            deep_fac2_c=deep_fac2_c,
            rho_fac_c=rho_fac_c,
            unit_deep_density_factors_verified=unit_deep_density_factors_verified,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        unavailable["ke_vol"] = str(exc)
    return {"values": values, "unavailable": unavailable}


def produce_regional_tier0(
    grid: object,
    grid_fields: Mapping[str, object],
    *,
    grid_fields_semantics_verified: bool = False,
    deep_fac2_c: torch.Tensor | None = None,
    rho_fac_c: torch.Tensor | None = None,
    unit_deep_density_factors_verified: bool = False,
) -> dict[str, dict[str, float | str]]:
    """Produce Tier-0 from a regional grid and its proven 21 base fields."""
    if getattr(grid, "face_count", 1) != 1:
        grid_fields = {}
    return _produce_tier0(
        grid,
        grid_fields,
        grid_fields_semantics_verified=grid_fields_semantics_verified,
        deep_fac2_c=deep_fac2_c,
        rho_fac_c=rho_fac_c,
        unit_deep_density_factors_verified=unit_deep_density_factors_verified,
    )


def _runtime_coriolis_fields(
    dynamics_modules: Iterable[object],
    *,
    ny: int,
    nx: int,
    dtype: torch.dtype,
    device: torch.device,
    extras_profile_g: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    profile_c: object | None = None
    profile_g: object | None = None
    for module in dynamics_modules:
        candidate_c = getattr(module, "coriolis_profile_c", None)
        candidate_g = getattr(module, "coriolis_profile_g", None)
        if candidate_c is not None:
            profile_c = candidate_c
        if candidate_g is not None:
            profile_g = candidate_g

    if profile_g is None and extras_profile_g is not None:
        # Translator-retained corner profile (ini_cori.F:77-78 construction,
        # emitted for flux-form assemblies whose runtime carries centres only).
        # Corner points are ny+1: a centre-length list here is not a corner
        # profile and must not be silently accepted.
        candidate = torch.as_tensor(extras_profile_g, dtype=dtype, device=device)
        if candidate.numel() != ny + 1:
            raise _Unavailable(
                f"retained corner Coriolis profile must have length {ny + 1}"
            )
        profile_g = candidate

    f_c = (
        None
        if profile_c is None
        else torch.as_tensor(profile_c, dtype=dtype, device=device)
    )
    f_g = (
        None
        if profile_g is None
        else torch.as_tensor(profile_g, dtype=dtype, device=device)
    )
    if f_c is not None and tuple(f_c.shape) != (ny,):
        raise _Unavailable(f"cell-centre Coriolis profile must have shape ({ny},)")
    if f_g is not None and tuple(f_g.shape) not in {(ny,), (ny + 1,)}:
        raise _Unavailable(
            f"corner Coriolis profile must have shape ({ny},) or ({ny + 1},)"
        )
    if f_c is None and f_g is None:
        raise _Unavailable("runtime Coriolis profiles are missing")

    if f_c is None:
        assert f_g is not None
        if f_g.numel() != ny + 1:
            raise _Unavailable("cell-centre Coriolis cannot be retained from a short corner profile")
        f_c = 0.5 * (f_g[:-1] + f_g[1:])
    if f_g is None:
        if ny < 2:
            raise _Unavailable("corner Coriolis cannot be retained from a one-row centre profile")
        delta = f_c[1:] - f_c[:-1]
        tolerance = 8.0 * torch.finfo(dtype).eps * torch.maximum(
            f_c.abs().max(), torch.ones((), dtype=dtype, device=device)
        )
        if not bool((delta - delta[0]).abs().le(tolerance).all().item()):
            raise _Unavailable("cell-centre Coriolis profile is not a verified linear beta plane")
        f_g = f_c - 0.5 * delta[0]

    return (
        f_c.view(ny, 1).expand(ny, nx),
        f_g[:ny].view(ny, 1).expand(ny, nx),
    )


def _spherical_coriolis_fields(
    dynamics_modules: Iterable[object],
    *,
    y_c: torch.Tensor,
    y_g: torch.Tensor,
    nx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retain spherical Coriolis fields after verifying the runtime profile."""
    profile_c: object | None = None
    profile_g: object | None = None
    for module in dynamics_modules:
        candidate_c = getattr(module, "coriolis_profile_c", None)
        candidate_g = getattr(module, "coriolis_profile_g", None)
        if candidate_c is not None:
            profile_c = candidate_c
        if candidate_g is not None:
            profile_g = candidate_g

    dtype = y_c.dtype
    device = y_c.device
    f_c = None if profile_c is None else torch.as_tensor(profile_c, dtype=dtype, device=device)
    f_g = None if profile_g is None else torch.as_tensor(profile_g, dtype=dtype, device=device)
    if f_c is not None and tuple(f_c.shape) != tuple(y_c.shape):
        raise _Unavailable(
            f"cell-centre Coriolis profile must have shape {tuple(y_c.shape)}"
        )
    if f_g is not None and tuple(f_g.shape) != tuple(y_g.shape):
        raise _Unavailable(
            f"corner Coriolis profile must have shape {tuple(y_g.shape)}"
        )
    if f_c is None and f_g is None:
        raise _Unavailable("runtime spherical Coriolis profiles are missing")
    for profile, label in ((f_c, "cell-centre"), (f_g, "corner")):
        if profile is not None and not bool(torch.isfinite(profile).all().item()):
            raise _Unavailable(f"runtime {label} Coriolis profile is non-finite")

    eps = torch.finfo(dtype).eps
    omega_samples: list[torch.Tensor] = []
    for profile, latitude in ((f_c, y_c), (f_g, y_g)):
        if profile is None:
            continue
        sine = torch.sin(torch.deg2rad(latitude))
        usable = sine.abs() > math.sqrt(eps)
        if bool(usable.any().item()):
            omega_samples.append(profile[usable] / (2.0 * sine[usable]))
    if not omega_samples:
        raise _Unavailable(
            "runtime spherical Coriolis profiles do not verify a rotation rate"
        )

    samples = torch.cat(omega_samples)
    omega = samples[0]
    omega_tolerance = 64.0 * eps * max(
        abs(float(omega.item())), torch.finfo(dtype).tiny
    )
    if not bool((samples - omega).abs().le(omega_tolerance).all().item()):
        raise _Unavailable("runtime spherical Coriolis profiles disagree on rotation rate")

    expected_c = 2.0 * omega * torch.sin(torch.deg2rad(y_c))
    expected_g = 2.0 * omega * torch.sin(torch.deg2rad(y_g))
    profile_tolerance = 64.0 * eps * max(
        abs(float(omega.item())), torch.finfo(dtype).tiny
    )
    if f_c is not None and not bool(
        torch.isclose(
            f_c, expected_c, rtol=64.0 * eps, atol=profile_tolerance
        ).all().item()
    ):
        raise _Unavailable("cell-centre Coriolis profile is not verified spherical")
    if f_g is not None and not bool(
        torch.isclose(
            f_g, expected_g, rtol=64.0 * eps, atol=profile_tolerance
        ).all().item()
    ):
        raise _Unavailable("corner Coriolis profile is not verified spherical")

    retained_c = expected_c if f_c is None else f_c
    retained_g = expected_g if f_g is None else f_g
    f_cos = 2.0 * omega * torch.cos(torch.deg2rad(y_c))
    return (
        retained_c.view(-1, 1).expand(-1, nx),
        retained_g[:-1].view(-1, 1).expand(-1, nx),
        f_cos.view(-1, 1).expand(-1, nx),
    )


def _regional_metric_fields(
    metrics: object,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Read MITgcm-named regional metrics without changing their staggering."""

    def retained(accessor_name: str, label: str) -> torch.Tensor:
        accessor = getattr(metrics, accessor_name, None)
        if not callable(accessor):
            raise _Unavailable(f"runtime grid metric accessor {accessor_name} is missing")
        return _float64_tensor(accessor(dtype, device), label)

    area = _float64_tensor(getattr(metrics, "cell_area", None), "RA")
    dx_f = _float64_tensor(getattr(metrics, "dx_center", None), "DXF")
    dy_f = _float64_tensor(getattr(metrics, "dy_center", None), "DYF")
    return {
        "DXC": retained("dxC_for", "DXC"),
        "DXF": dx_f,
        "DXG": retained("dxG_for", "DXG"),
        "DXV": retained("dxV_for", "DXV"),
        "DYC": retained("dyC_for", "DYC"),
        "DYF": dy_f,
        "DYG": retained("dyG_for", "DYG"),
        "DYU": retained("dyU_for", "DYU"),
        "RA": area,
        "RAW": retained("rAw_for", "RAW"),
        "RAS": retained("rAs_for", "RAS"),
        "RAZ": retained("rAz_for", "RAZ"),
    }


def _cartesian_spacing(
    field: torch.Tensor,
    *,
    ny: int,
    nx: int,
    axis: str,
) -> torch.Tensor:
    if field.ndim != 2 or field.shape[0] < ny or field.shape[1] < nx:
        raise _Unavailable(
            f"Cartesian {axis}-spacing cannot provide physical shape ({ny}, {nx})"
        )
    physical = field[:ny, :nx]
    if axis == "x":
        spacing = physical[0, :]
        aligned = spacing.view(1, nx).expand(ny, nx)
    else:
        spacing = physical[:, 0]
        aligned = spacing.view(ny, 1).expand(ny, nx)
    if not torch.equal(physical, aligned):
        raise _Unavailable(f"runtime Cartesian {axis}-spacing is not axis-aligned")
    if not bool((spacing > 0.0).all().item()):
        raise _Unavailable(f"runtime Cartesian {axis}-spacing must be positive")
    return spacing


def _prefix_coordinates(
    origin: float, spacing: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.cat(
        (torch.zeros_like(spacing[:1]), torch.cumsum(spacing[:-1], dim=0))
    )
    corners = origin + offsets
    return corners, corners + 0.5 * spacing


def regional_runtime_grid_fields(
    grid: object,
    run_cfg: object,
    dynamics_modules: Iterable[object],
) -> dict[str, torch.Tensor]:
    """Retain verified Cartesian or spherical-polar regional grid fields."""
    extras = getattr(getattr(run_cfg, "case", None), "extras", None)
    if not isinstance(extras, Mapping):
        raise _Unavailable("case.extras is missing")
    mitcase = extras.get("mitcase")
    if not isinstance(mitcase, Mapping):
        raise _Unavailable("case.extras.mitcase is missing")
    family = mitcase.get("grid_family")
    if family not in {"cartesian", "spherical_polar"}:
        raise _Unavailable(f"unsupported regional Tier-0 grid family: {family!r}")
    if extras.get("grid_mode") != family:
        raise _Unavailable(
            "case.extras.grid_mode does not match the verified MITgcm grid family"
        )
    if getattr(grid, "face_count", 1) != 1:
        raise _Unavailable("regional Tier-0 field retention does not accept cube grids")

    ny = int(getattr(grid, "ny"))
    nx = int(getattr(grid, "nx"))
    metrics = getattr(grid, "metrics", None)
    if metrics is None:
        raise _Unavailable("runtime grid metrics are missing")
    area = _float64_tensor(getattr(metrics, "cell_area", None), "rA")
    dtype = area.dtype
    device = area.device
    metric_fields = _regional_metric_fields(metrics, dtype=dtype, device=device)

    if family == "cartesian":
        try:
            xg_origin = float(mitcase["xg_origin_m"])
            yg_origin = float(mitcase["yg_origin_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise _Unavailable("verified Cartesian grid origins are missing") from exc
        if not math.isfinite(xg_origin) or not math.isfinite(yg_origin):
            raise _Unavailable("verified Cartesian grid origins must be finite")
        del_x = _cartesian_spacing(metric_fields["DXF"], ny=ny, nx=nx, axis="x")
        del_y = _cartesian_spacing(metric_fields["DYF"], ny=ny, nx=nx, axis="y")
        x_g, x_c = _prefix_coordinates(xg_origin, del_x)
        y_g, y_c = _prefix_coordinates(yg_origin, del_y)
        f_c, f_g = _runtime_coriolis_fields(
            dynamics_modules,
            ny=ny,
            nx=nx,
            dtype=dtype,
            device=device,
            extras_profile_g=mitcase.get("coriolis_profile_g"),
        )
        f_cos = torch.zeros_like(area)
    else:
        grid_cfg = getattr(run_cfg, "grid", None)
        try:
            cfg_nx = int(getattr(grid_cfg, "nx"))
            cfg_ny = int(getattr(grid_cfg, "ny"))
            resolution = float(getattr(grid_cfg, "resolution_deg"))
            center_lon = float(getattr(grid_cfg, "center_lon"))
            center_lat = float(getattr(grid_cfg, "center_lat"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise _Unavailable("verified spherical grid coordinates are missing") from exc
        if (cfg_nx, cfg_ny) != (nx, ny):
            raise _Unavailable("runtime and configured spherical grid shapes disagree")
        if (
            not all(
                math.isfinite(value)
                for value in (resolution, center_lon, center_lat)
            )
            or resolution <= 0.0
        ):
            raise _Unavailable(
                "verified spherical grid coordinates must be finite and positive"
            )
        try:
            xg_origin = float(
                mitcase.get("xg_origin_deg", center_lon - 0.5 * nx * resolution)
            )
            yg_origin = float(
                extras.get(
                    "yg_origin",
                    mitcase.get(
                        "yg_origin_deg", center_lat - 0.5 * ny * resolution
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise _Unavailable("verified spherical grid origins are invalid") from exc
        if not math.isfinite(xg_origin) or not math.isfinite(yg_origin):
            raise _Unavailable("verified spherical grid origins must be finite")
        del_x = torch.full((nx,), resolution, dtype=dtype, device=device)
        del_y = torch.full((ny,), resolution, dtype=dtype, device=device)
        x_g, x_c = _prefix_coordinates(xg_origin, del_x)
        y_g, y_c = _prefix_coordinates(yg_origin, del_y)
        f_c, f_g, f_cos = _spherical_coriolis_fields(
            dynamics_modules,
            y_c=y_c,
            y_g=torch.cat((y_g, y_g[-1:] + resolution)),
            nx=nx,
        )

    xg = x_g.view(1, nx).expand(ny, nx)
    yg = y_g.view(ny, 1).expand(ny, nx)
    angle_cs = torch.ones_like(area)
    angle_sn = torch.zeros_like(area)
    return {
        "XC": x_c.view(1, nx).expand(ny, nx),
        "XG": xg,
        **{key: metric_fields[key] for key in ("DXC", "DXF", "DXG", "DXV")},
        "YC": y_c.view(ny, 1).expand(ny, nx),
        "YG": yg,
        **{key: metric_fields[key] for key in ("DYC", "DYF", "DYG", "DYU")},
        **{key: metric_fields[key] for key in ("RA", "RAW", "RAS", "RAZ")},
        "AngleCS": angle_cs,
        "AngleSN": angle_sn,
        "fCori": f_c,
        "fCoriG": f_g,
        "fCoriCos": f_cos,
    }


def produce_regional_runtime_tier0(
    grid: object,
    run_cfg: object,
    dynamics_modules: Iterable[object],
) -> dict[str, dict[str, float | str]]:
    """Produce Tier-0 from the field objects retained by a regional runtime."""
    try:
        fields = regional_runtime_grid_fields(grid, run_cfg, dynamics_modules)
    except (AttributeError, TypeError, ValueError) as exc:
        result = produce_regional_tier0(
            grid,
            {},
            unit_deep_density_factors_verified=True,
        )
        result["unavailable"].update({key: str(exc) for key in GRID_KEYS})
        return result
    return produce_regional_tier0(
        grid,
        fields,
        grid_fields_semantics_verified=True,
        unit_deep_density_factors_verified=True,
    )


def produce_cube_tier0(
    grid: object,
    coriolis_fields: Mapping[str, object],
    *,
    grid_fields_semantics_verified: bool = False,
    deep_fac2_c: torch.Tensor | None = None,
    rho_fac_c: torch.Tensor | None = None,
    unit_deep_density_factors_verified: bool = False,
) -> dict[str, dict[str, float | str]]:
    """Produce Tier-0 from cube ``face_geometry`` plus three Coriolis fields."""
    fields: dict[str, object] = {}
    if getattr(grid, "face_count", None) == 6:
        face_geometry = getattr(getattr(grid, "metrics", None), "face_geometry", None)
        if isinstance(face_geometry, Mapping):
            fields.update(face_geometry)
        fields.update(coriolis_fields)
    return _produce_tier0(
        grid,
        fields,
        grid_fields_semantics_verified=grid_fields_semantics_verified,
        deep_fac2_c=deep_fac2_c,
        rho_fac_c=rho_fac_c,
        unit_deep_density_factors_verified=unit_deep_density_factors_verified,
    )


__all__ = [
    "produce_cube_tier0",
    "produce_regional_runtime_tier0",
    "produce_regional_tier0",
    "regional_runtime_grid_fields",
]
