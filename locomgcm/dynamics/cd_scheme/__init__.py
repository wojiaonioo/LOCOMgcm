"""MITgcm CD scheme (``pkg/cd_code``): Coriolis from a D-grid velocity.

``kernels`` holds the tensor-only statements of ``cd_code_scheme.F``,
``state`` the cross-step D-grid velocity and lagged fields with their restart
codec, ``module`` the registered ``cd_scheme`` module, its admission and the
adapter that exchanges halos between the kernel stages.

Signs are MITgcm's and LOCOMgcm's alike for every field used here: u east,
v north, eta up (m).  ``vVelD`` lives on U points and is stored with the U
shape ``[B, nz, ny, nx+1]``; ``uVelD`` lives on V points with the V shape
``[B, nz, ny+1, nx]``.  Velocities are m s-1, tendencies m s-2.
"""

from .module import (
    CD_CODE_OPTIONS, CD_MOMENTUM_FORMS, CD_STEPPING_MODES, CdScheme, admit_cd_scheme, cd_scheme_of,
)
from .state import CdState, decode_cd_state, encode_cd_state

__all__ = [
    "CD_CODE_OPTIONS",
    "CD_MOMENTUM_FORMS",
    "CD_STEPPING_MODES",
    "CdScheme",
    "CdState",
    "admit_cd_scheme",
    "cd_scheme_of",
    "decode_cd_state",
    "encode_cd_state",
]
