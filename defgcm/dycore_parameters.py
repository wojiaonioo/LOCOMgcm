"""Nonhydrostatic CG3D controls and the MITgcm 3-D Coriolis control resolution.

``NonhydrostaticParams`` carries the stepper's nonHydrostatic and cg3d settings;
``resolve_3d_coriolis_controls`` turns MITgcm's select3dCoriScheme /
use3dCoriolis / selectCoriMap / fPrime into the effective scheme, map and
fCoriCos for one grid geometry.  All values are dimensionless selectors except
``f_prime`` (s^-1) and the CG3D residual targets.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from defgcm.constants import ROTATION_PERIOD_EARTH


@dataclass(frozen=True)
class NonhydrostaticParams:
    """nonHydrostatic + cg3d 3D pressure solver parameters."""

    nonhydrostatic: bool = False
    cg3d_max_iters: int = 150
    cg3d_target_residual: float = 1e-7
    cg3d_target_res_w_unit: float = -1.0
    nh_am2: float = 1.0
    select_3d_cori_scheme: int = 1
    select_cori_map: int = -1
    f_prime: float = 0.0
    #: MITgcm rotationPeriod [s] (set_defaults.F:115); the latitude form of
    #: fCoriCos in MOM_W_CORIOLIS_NH uses its omega (ini_parms.F:487-490).
    rotation_period: float = ROTATION_PERIOD_EARTH


#: MITgcm selectCoriMap values accepted here; the configuration value is the MITgcm
#: code (ini_cori.F: 0 f-plane, 1 beta-plane, 2 spherical), and -1 is UNSET, the grid
#: default.  Map 3 (fCori read from file) is not accepted.
SELECT_CORI_MAPS = (-1, 0, 1, 2)


@dataclass(frozen=True)
class Resolved3DCoriolis:
    """MIT-effective 3-D Coriolis controls for one grid geometry."""

    select_3d_cori_scheme: int
    select_cori_map: int
    f_cori_cos: float | None


def resolve_3d_coriolis_controls(
    *,
    nonhydrostatic: bool,
    select_3d_cori_scheme: int | None,
    use_3d_coriolis: bool = True,
    select_cori_map: int = -1,
    f_prime: float = 0.0,
    grid_is_spherical: bool,
    use_coriolis: bool = True,
) -> Resolved3DCoriolis:
    """Resolve MITgcm's backward-compatible selector and map-dependent gate.

    ``None`` represents MITgcm's ``UNSET_I`` selector.  Map ``-1`` selects the
    grid default: Beta-plane for Cartesian grids and spherical otherwise.
    Coriolis map 3 requires an externally supplied field and is intentionally
    rejected because LOCOMgcm cannot express that runtime input here.
    """
    if type(nonhydrostatic) is not bool:
        raise TypeError("nonhydrostatic must be bool")
    if type(use_3d_coriolis) is not bool:
        raise TypeError("use_3d_coriolis must be bool")
    if type(grid_is_spherical) is not bool:
        raise TypeError("grid_is_spherical must be bool")
    if type(use_coriolis) is not bool:
        raise TypeError("use_coriolis must be bool")
    from defgcm.dynamics.vector_invariant_momentum import CORI_3D_SCHEMES

    cori_3d_codes = tuple(code for _name, code in CORI_3D_SCHEMES)
    if select_3d_cori_scheme is not None and (
        isinstance(select_3d_cori_scheme, bool)
        or not isinstance(select_3d_cori_scheme, int)
        or select_3d_cori_scheme not in cori_3d_codes
    ):
        raise ValueError(
            f"select_3d_cori_scheme must be None or one of {cori_3d_codes}, "
            f"got {select_3d_cori_scheme!r}"
        )
    if isinstance(select_cori_map, bool) or not isinstance(select_cori_map, int):
        raise TypeError("select_cori_map must be int")
    if select_cori_map not in SELECT_CORI_MAPS:
        raise ValueError(
            f"select_cori_map must be one of {SELECT_CORI_MAPS}; map 3 requires "
            f"an external fCori field, got {select_cori_map}"
        )
    if isinstance(f_prime, bool) or not isinstance(f_prime, (int, float)):
        raise TypeError("f_prime must be numeric")
    resolved_f_prime = float(f_prime)
    if not math.isfinite(resolved_f_prime):
        raise ValueError(f"f_prime must be finite, got {resolved_f_prime}")

    if select_3d_cori_scheme is None:
        requested_scheme = 1 if use_3d_coriolis else 0
    else:
        requested_scheme = select_3d_cori_scheme
        if requested_scheme != 0 and not use_3d_coriolis:
            raise ValueError(
                "select_3d_cori_scheme conflicts with use_3d_coriolis=False"
            )

    resolved_map = select_cori_map
    if resolved_map == -1:
        resolved_map = 2 if grid_is_spherical else 1

    effective_scheme = requested_scheme
    if not nonhydrostatic or not use_coriolis:
        effective_scheme = 0
    if resolved_map in (0, 1) and resolved_f_prime == 0.0:
        effective_scheme = 0

    return Resolved3DCoriolis(
        select_3d_cori_scheme=effective_scheme,
        select_cori_map=resolved_map,
        f_cori_cos=(resolved_f_prime if resolved_map in (0, 1) else None),
    )

