"""Cross-step state of the CD scheme and its restart codec.

``CdState`` holds MITgcm's ``CD_CODE_VARS.h`` fields after a step:
``u_d`` (uVelD, V points ``[B, nz, ny+1, nx]``), ``v_d`` (vVelD, U points
``[B, nz, ny, nx+1]``), ``eta_nm1`` (etaNm1, cell centres ``[B, ny, nx]``, m,
up positive), ``u_nm1`` and ``v_nm1`` (uNM1, vNM1, the C-grid velocity of the
step that produced the state).  Dry faces hold zero D-grid velocity.  The
stepper owns the one live copy (``ImplicitFreeSurfaceStepper._cd_state``);
``None`` there is a fresh run, MITgcm's zero initial state at ``myIter=0``.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch

#: Restart key of the CD state (``locomgcm.checkpoint.state.RESTART_STATE_KEYS_V2``).
CD_RESTART_KEY = "_cd_scheme"


@dataclass(frozen=True)
class CdState:
    u_d: torch.Tensor
    v_d: torch.Tensor
    eta_nm1: torch.Tensor
    u_nm1: torch.Tensor
    v_nm1: torch.Tensor


def encode_cd_state(state: CdState | None):
    if state is None:
        return {}
    return {CD_RESTART_KEY: {item.name: getattr(state, item.name) for item in fields(CdState)}}


def decode_cd_state(payload):
    if CD_RESTART_KEY not in payload:
        return None
    return CdState(**payload[CD_RESTART_KEY])
