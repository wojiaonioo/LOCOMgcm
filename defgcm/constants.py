"""Centralised physical constants shared by LOCOMgcm core and experiment plumbing.

Single source of truth so MIT-parity values are not hardcoded in scattered call sites.
"""
from __future__ import annotations

import math

# Earth angular velocity, MIT default convention.
#
# MITgcm sets ``rotationPeriod = 86164.0`` (integer seconds, ``set_defaults.F:115``)
# and derives ``omega = 2 * pi / rotationPeriod`` (``ini_parms.F:487-490``).
# This expression preserves the MITgcm rounding at the Coriolis term.
#: Earth's sidereal rotation period in seconds; MITgcm ``rotationPeriod`` default
#: (``set_defaults.F:115``).
ROTATION_PERIOD_EARTH: float = 86164.0


def omega_from_rotation_period(rotation_period: float) -> float:
    """MITgcm ``omega`` when the namelist leaves ``omega`` unset.

    ``ini_parms.F:487-490``: ``2*pi/rotationPeriod``, and 0 for
    ``rotationPeriod = 0`` (no rotation).
    """
    if rotation_period == 0.0:
        return 0.0
    return 2.0 * math.pi / rotation_period


OMEGA_EARTH: float = omega_from_rotation_period(ROTATION_PERIOD_EARTH)

# Boussinesq reference density rhoConst [kg/m^3] — single default for the whole
# surface-forcing / EXF subsystem.
#
# Every EXF default reads this one value, so a call site that does not pass
# ``rho_const`` gets the same density as its neighbours and as the EOS/HPG
# ``reference_density`` default.  Every production case sets ``rhoConst`` explicitly (all
# currently 1025), so this is only the unset default, not a hardcoded runtime value.
#
# MIT keeps ``rhoConstFresh`` defaulting to ``rhoConst`` (ini_parms.F), so the
# freshwater density default reuses this same constant rather than a third value.
RHO_CONST_DEFAULT: float = 1025.0
