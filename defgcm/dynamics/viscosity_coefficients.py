"""MITgcm MOM_CALC_VISC coefficients: Leith, Smagorinsky, base viscosity and grid limits.

Kernels only.  Inputs are the gradient and strain measures and the harmonic
length scales ``L2``/``L3`` at D (cell centre, ``[..., ny, nx]``) and Z (corner,
``[..., ny+1, nx+1]``) points, Python-float coefficients from the namelist, and
the grid+dt scales of the floor/cap.  Outputs are ``viscAh_D/Z`` (m^2/s) and
``viscA4_D/Z`` (m^4/s) in the measures' dtype.  Which assembly runs (max-norm or
divergence/full Leith, harmonic or biharmonic, clamped or not) is decided by the
caller; the kernels never read the grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ViscosityMeasures:
    """Gradient and strain measures with the length scales, shared by the
    harmonic and the biharmonic assembly.

    The ``*_sq`` and divergence entries are ``None`` when the producer's gate
    was shut; ``div_measures_present`` records that decision.
    """

    vrt_d_max: torch.Tensor
    grd_vrt_d_sq: torch.Tensor | None
    grd_div_d_max: torch.Tensor | None
    grd_div_d_sq: torch.Tensor | None
    smag_d: torch.Tensor
    l2_d: torch.Tensor
    l3_d: torch.Tensor
    vrt_z_max: torch.Tensor
    grd_vrt_z_sq: torch.Tensor | None
    grd_div_z_max: torch.Tensor | None
    grd_div_z_sq: torch.Tensor | None
    smag_z: torch.Tensor
    l2_z: torch.Tensor
    l3_z: torch.Tensor
    div_measures_present: bool


def _require_div_measures(m: ViscosityMeasures, needed: bool, what: str) -> None:
    """Refuse measures built with the divergence-gradient gate shut.

    ``_leith_grad_measures`` skips ``divDx/divDy``, both point-divergence
    measures and both ``grd_vrt_*_sq`` when no consumer wants them, and records
    that decision under ``div_measures_present``.  Assemblies that DO want them
    check it here, so measures produced by one ``HorizontalViscosity`` and
    consumed by another with different switches fails with the reason rather
    than with a ``NoneType`` multiply inside a Leith contribution kernel.
    """
    if not needed:
        return
    if m.div_measures_present:
        return
    raise ValueError(
        f"{what} viscosity assembly needs the divergence-gradient measures, but "
        "these measures were built with the gate shut "
        "(_needs_leith_sq_and_div_measures() was False for the producing "
        "HorizontalViscosity).  Build the measures with the SAME instance that "
        "assembles them, or with one whose use_full_leith / visc_c2_leith_d / "
        "visc_c4_* switches make the gate open."
    )


def _safe_sqrt_zero_grad(x: torch.Tensor) -> torch.Tensor:
    safe = torch.where(x > 0, x, torch.ones_like(x))
    return torch.where(x > 0, torch.sqrt(safe), torch.zeros_like(x))


def _leith_contribution_full(
    grd_vrt_sq: torch.Tensor,
    grd_div_sq: torch.Tensor,
    c_vort: torch.Tensor,
    c_div: torch.Tensor,
    l_pow: torch.Tensor,
    *,
    full_const: float,
    pi: torch.Tensor,
) -> torch.Tensor:
    """MIT full-Leith contribution (``useFullLeith=.TRUE.``):
    sqrt(fv*grdVrt_sq + fd*grdDiv_sq)*L with the ^6 coefficient powers (the
    const carries MIT's 0.015625=1/64 for biharmonic, which factors out of the
    sqrt as 1/8)."""
    cv = c_vort / pi
    cd = c_div / pi
    fv = cv.pow(6) * full_const
    fd = cd.pow(6) * full_const
    return _safe_sqrt_zero_grad(fv * grd_vrt_sq + fd * grd_div_sq) * l_pow


def _leith_contribution_maxnorm(
    grd_vrt_max: torch.Tensor,
    grd_div_max: torch.Tensor,
    c_vort: torch.Tensor,
    c_div: torch.Tensor,
    l_pow: torch.Tensor,
    *,
    approx_const: float,
    pi: torch.Tensor,
) -> torch.Tensor:
    """MIT max-norm Leith contribution (``useFullLeith=.FALSE.``):
    (fv*grdVrt_max + fd*grdDiv_max)*L with the ^3 powers (const 1 / 0.125)."""
    cv = c_vort / pi
    cd = c_div / pi
    fv = cv.pow(3) * approx_const
    fd = cd.pow(3) * approx_const
    return (fv * grd_vrt_max + fd * grd_div_max) * l_pow


def _leith_vorticity_maxnorm_contribution(
    leith_factor: torch.Tensor,
    vrt_max: torch.Tensor,
    l_pow: torch.Tensor,
) -> torch.Tensor:
    """Legacy MIT Leith vorticity-only max-norm contribution."""
    return leith_factor * vrt_max * l_pow


def _smag_harmonic_viscosity_contribution(
    smag_factor: torch.Tensor,
    smag: torch.Tensor,
    l2: torch.Tensor,
) -> torch.Tensor:
    """MIT harmonic Smagorinsky viscosity contribution."""
    return smag_factor * smag * l2


def _add_harmonic_base_viscosity(
    viscAh_D: torch.Tensor,
    viscAh_Z: torch.Tensor,
    base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add constant MIT harmonic base viscosity to D/Z coefficients."""
    viscAh_D = viscAh_D + base
    viscAh_Z = viscAh_Z + base
    return viscAh_D, viscAh_Z


def _harmonic_grid_rdt_scale(l2: torch.Tensor, dt: float) -> torch.Tensor:
    """MIT harmonic grid floor/cap scale ``0.25 * L2 / dt``."""
    return (0.25 / dt) * l2


def _grid_floor(
    visc_D: torch.Tensor,
    visc_Z: torch.Tensor,
    scale_d: torch.Tensor,
    scale_z: torch.Tensor,
    grid_min: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT grid floor ``MAX(visc, gridMin*scale)`` at D and Z points; ``scale``
    is ``L2rdt`` (harmonic) or ``L4rdt`` (biharmonic)."""
    visc_D = torch.maximum(visc_D, grid_min * scale_d)
    visc_Z = torch.maximum(visc_Z, grid_min * scale_z)
    return visc_D, visc_Z


def _grid_cap(
    visc_D: torch.Tensor,
    visc_Z: torch.Tensor,
    scale_d: torch.Tensor,
    scale_z: torch.Tensor,
    grid_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT grid cap ``MIN(visc, gridMax*scale)`` at D and Z points."""
    visc_D = torch.minimum(visc_D, grid_max * scale_d)
    visc_Z = torch.minimum(visc_Z, grid_max * scale_z)
    return visc_D, visc_Z


def _w_viscosity_from_center_viscosity(visc_center: torch.Tensor) -> torch.Tensor:
    """Map MIT center viscosity to Nr-sized W-momentum storage."""
    visc_w = torch.empty_like(visc_center)
    visc_w[:, 0] = visc_center[:, 0]
    if visc_center.shape[1] > 1:
        visc_w[:, 1:] = 0.5 * (visc_center[:, :-1] + visc_center[:, 1:])
    return visc_w


def _biharmonic_length_powers(l2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT biharmonic Leith/Smag length powers from L2."""
    l5 = l2.pow(2.5)
    l4 = l2.pow(2)
    return l5, l4


def _biharmonic_grid_rdt_scale(l2: torch.Tensor, dt: float) -> torch.Tensor:
    """MIT biharmonic grid floor/cap scale ``0.03125 * L2^2 / dt``."""
    return (0.03125 / dt) * l2.pow(2)


def _add_biharmonic_base_viscosity(
    viscA4_D: torch.Tensor,
    viscA4_Z: torch.Tensor,
    base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add constant MIT biharmonic base viscosity to D/Z coefficients."""
    viscA4_D = base + viscA4_D
    viscA4_Z = base + viscA4_Z
    return viscA4_D, viscA4_Z


def _smag_biharmonic_viscosity_contribution(
    smag_factor: torch.Tensor,
    l4: torch.Tensor,
    smag: torch.Tensor,
) -> torch.Tensor:
    """MIT biharmonic Smagorinsky viscosity contribution."""
    return smag_factor * l4 * smag


def _harmonic_leith_vorticity(
    m: ViscosityMeasures,
    *,
    visc_c2_leith: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Harmonic Leith at D and Z from the max-norm vorticity gradient only
    (``useFullLeith=.FALSE.``, ``viscC2LeithD=0``)."""
    ref = m.l2_d
    pi = ref.new_tensor(torch.pi)
    # Legacy path — byte-identical to pre-completion behaviour.
    leith2fac = (ref.new_tensor(visc_c2_leith) / pi) ** 3
    leith_d = _leith_vorticity_maxnorm_contribution(leith2fac, m.vrt_d_max, m.l3_d)
    leith_z = _leith_vorticity_maxnorm_contribution(leith2fac, m.vrt_z_max, m.l3_z)
    return leith_d, leith_z


def _harmonic_leith_maxnorm(
    m: ViscosityMeasures,
    *,
    visc_c2_leith: float,
    visc_c2_leith_d: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Harmonic Leith at D and Z with the divergence term, max-norm form."""
    ref = m.l2_d
    pi = ref.new_tensor(torch.pi)
    c2 = ref.new_tensor(visc_c2_leith)
    c2d = ref.new_tensor(visc_c2_leith_d)
    leith_d = _leith_contribution_maxnorm(
        m.vrt_d_max, m.grd_div_d_max, c2, c2d, m.l3_d, approx_const=1.0, pi=pi,
    )
    leith_z = _leith_contribution_maxnorm(
        m.vrt_z_max, m.grd_div_z_max, c2, c2d, m.l3_z, approx_const=1.0, pi=pi,
    )
    return leith_d, leith_z


def _harmonic_leith_full(
    m: ViscosityMeasures,
    *,
    visc_c2_leith: float,
    visc_c2_leith_d: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Harmonic Leith at D and Z with the divergence term, full-Leith form."""
    ref = m.l2_d
    pi = ref.new_tensor(torch.pi)
    c2 = ref.new_tensor(visc_c2_leith)
    c2d = ref.new_tensor(visc_c2_leith_d)
    leith_d = _leith_contribution_full(
        m.grd_vrt_d_sq, m.grd_div_d_sq, c2, c2d, m.l3_d, full_const=1.0, pi=pi,
    )
    leith_z = _leith_contribution_full(
        m.grd_vrt_z_sq, m.grd_div_z_sq, c2, c2d, m.l3_z, full_const=1.0, pi=pi,
    )
    return leith_d, leith_z


def _harmonic_viscosity_with_smag(
    leith_d: torch.Tensor,
    leith_z: torch.Tensor,
    m: ViscosityMeasures,
    *,
    visc_c2_smag: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``viscAh_D/Z`` = Leith + harmonic Smagorinsky, without base or limits."""
    ref = m.l2_d
    pi = ref.new_tensor(torch.pi)
    smag2fac = (ref.new_tensor(visc_c2_smag) / pi) ** 2
    viscAh_D = leith_d + _smag_harmonic_viscosity_contribution(
        smag2fac, m.smag_d, m.l2_d,
    )
    viscAh_Z = leith_z + _smag_harmonic_viscosity_contribution(
        smag2fac, m.smag_z, m.l2_z,
    )
    return viscAh_D, viscAh_Z


def _biharmonic_leith_maxnorm(
    m: ViscosityMeasures,
    l5_d: torch.Tensor,
    l5_z: torch.Tensor,
    *,
    visc_c4_leith: float,
    visc_c4_leith_d: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Biharmonic Leith at D and Z, max-norm form: leith4fac 0.125*(c4/pi)^3
    times ``L5 = L2^2.5`` (``mom_calc_visc.F`` biharmonic block)."""
    ref = m.l2_d
    pi = ref.new_tensor(torch.pi)
    c4 = ref.new_tensor(visc_c4_leith)
    c4d = ref.new_tensor(visc_c4_leith_d)
    leith4_d = _leith_contribution_maxnorm(
        m.vrt_d_max, m.grd_div_d_max, c4, c4d, l5_d, approx_const=0.125, pi=pi,
    )
    leith4_z = _leith_contribution_maxnorm(
        m.vrt_z_max, m.grd_div_z_max, c4, c4d, l5_z, approx_const=0.125, pi=pi,
    )
    return leith4_d, leith4_z


def _biharmonic_leith_full(
    m: ViscosityMeasures,
    l5_d: torch.Tensor,
    l5_z: torch.Tensor,
    *,
    visc_c4_leith: float,
    visc_c4_leith_d: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Biharmonic Leith at D and Z, full-Leith form: 0.015625*(c4/pi)^6 under
    the square root, times ``L5``."""
    ref = m.l2_d
    pi = ref.new_tensor(torch.pi)
    c4 = ref.new_tensor(visc_c4_leith)
    c4d = ref.new_tensor(visc_c4_leith_d)
    leith4_d = _leith_contribution_full(
        m.grd_vrt_d_sq, m.grd_div_d_sq, c4, c4d, l5_d, full_const=0.015625, pi=pi,
    )
    leith4_z = _leith_contribution_full(
        m.grd_vrt_z_sq, m.grd_div_z_sq, c4, c4d, l5_z, full_const=0.015625, pi=pi,
    )
    return leith4_d, leith4_z


def _biharmonic_viscosity_with_smag(
    leith4_d: torch.Tensor,
    leith4_z: torch.Tensor,
    m: ViscosityMeasures,
    l4_d: torch.Tensor,
    l4_z: torch.Tensor,
    *,
    visc_c4_smag: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``viscA4_D/Z`` = Leith4 + biharmonic Smagorinsky (smag4fac =
    0.125*(c4smag/pi)^2 times ``L2^2``), without base or limits."""
    ref = m.l2_d
    pi = ref.new_tensor(torch.pi)
    smag4fac = 0.125 * (ref.new_tensor(visc_c4_smag) / pi) ** 2
    viscA4_D = leith4_d + _smag_biharmonic_viscosity_contribution(
        smag4fac, l4_d, m.smag_d,
    )
    viscA4_Z = leith4_z + _smag_biharmonic_viscosity_contribution(
        smag4fac, l4_z, m.smag_z,
    )
    return viscA4_D, viscA4_Z
