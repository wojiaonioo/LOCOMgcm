"""Spherical Coriolis geometry and 3-D scheme resolution.

Latitude rows are in degrees at centres [ny] and corners [ny+1]; f is signed
2*Omega*sin(latitude) in s^-1, evaluated through float64 host profiles.
Unmasked centre metrics use area in m2 and widths in m, at their input dtype.
"""
from __future__ import annotations

from typing import Any

import torch

from locomgcm.dycore_parameters import (
    NonhydrostaticParams,
    Resolved3DCoriolis,
    resolve_3d_coriolis_controls,
)


#: Relative agreement that marks a corner Coriolis profile as the spherical map
#: of the grid's corner latitudes.  Rounding differences sit near 1e-16, while an
#: f-plane or beta-plane profile departs from the sphere by the curvature term,
#: above 1e-8 of f even across one 0.03-degree row.
_SPHERICAL_PROFILE_RTOL = 1.0e-12


def spherical_coriolis(lat_deg: float, omega: float) -> float:
    """``2*Omega*sin(lat)``, spelled as the ``$coriolis_profile_g/_c`` sentinels."""
    import numpy as np

    return float(2.0 * omega * np.sin(np.deg2rad(lat_deg)))


def spherical_centre_coriolis(
    profile_g: Any,
    *,
    centre: tuple[float, ...] | None,
    corner: tuple[float, ...] | None,
    omega: float,
) -> list[float] | None:
    """KPP's cell-centre f on a spherical grid, or ``None`` to keep the corner average.

    ``2*Omega*sin(yC)`` at the grid's centre latitudes (``MetricSet.yC_row``) when
    the grid carries latitude rows and the momentum's corner profile is, to
    rounding, the spherical map at its corner latitudes (``yG_row``), i.e. the
    momentum runs MITgcm's selectCoriMap=2.  On a Cartesian grid, or with an
    f-plane or beta-plane profile on any grid, the corner average is already the
    centre value of the map the momentum runs.
    """
    if centre is None or corner is None:
        return None
    given = torch.as_tensor(profile_g, dtype=torch.float64).detach().cpu().reshape(-1).tolist()
    expected = [spherical_coriolis(lat, omega) for lat in corner]
    if len(given) != len(expected):
        return None
    scale = max(abs(value) for value in expected)
    if max(abs(g - e) for g, e in zip(given, expected)) > _SPHERICAL_PROFILE_RTOL * scale:
        return None
    return [spherical_coriolis(lat, omega) for lat in centre]


def resolve_3d_coriolis(
    params: NonhydrostaticParams,
    *,
    cell_area: torch.Tensor,
    dx_center: torch.Tensor,
    dy_center: torch.Tensor,
) -> Resolved3DCoriolis:
    """Resolve the controls that MITgcm would deliver after ``SET_PARMS``."""
    # Rectangular metrics construct rA as dx*dy exactly.  Spherical-polar
    # metrics use the finite-latitude band area and are not bitwise equal to
    # that product.  This remains valid for scalar and vector spacing.
    grid_is_spherical = not torch.equal(
        cell_area,
        dx_center * dy_center,
    )
    if params.select_cori_map == 2 and not grid_is_spherical:
        raise ValueError(
            "select_cori_map=2 requires spherical-polar grid metrics; "
            "Cartesian coordinates cannot supply MITgcm's degree-valued yC"
        )
    return resolve_3d_coriolis_controls(
        nonhydrostatic=params.nonhydrostatic,
        select_3d_cori_scheme=params.select_3d_cori_scheme,
        select_cori_map=params.select_cori_map,
        f_prime=params.f_prime,
        grid_is_spherical=grid_is_spherical,
    )
