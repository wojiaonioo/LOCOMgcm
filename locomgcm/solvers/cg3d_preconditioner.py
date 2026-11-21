"""Vertical-line LU preconditioner for CG3D.

The preconditioner consumes normalized scalar CG3D coefficients only; no
``w``/``wVel`` sign convention is involved here.
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from locomgcm.accel.tensor_context import forward_mode_active

from .cg3d_operator import CG3DOperator


class CG3DPreconditioner(NamedTuple):
    zMC: torch.Tensor
    zML: torch.Tensor
    zMU: torch.Tensor


def build_cg3d_preconditioner(operator: CG3DOperator) -> CG3DPreconditioner:
    """Build MITgcm ``INI_CG3D`` vertical-line Thomas factors."""
    aC = operator.aC3d
    aV = operator.aV3d
    if aC.ndim != 3:
        raise ValueError(f"aC3d must be [nz,ny,nx], got {tuple(aC.shape)}")
    if aV.shape != aC.shape:
        raise ValueError("aV3d must match aC3d shape")

    nz = aC.shape[0]
    wet_diag = aC != 0
    # The downward sweep builds each level out of place and stacks at the end.
    # Writing it as ``zMC[k] = ...`` into a preallocated tensor bumps the version
    # counter of the very slice the next level reads, so the coefficient VJP is
    # rejected eagerly and -- under ``checkpoint`` recompute, where that guard no
    # longer fires -- comes back silently wrong.  Arithmetic is unchanged.
    base_C = torch.where(wet_diag, aC, torch.ones_like(aC))
    zML = torch.where(wet_diag, aV, torch.zeros_like(aV))
    if nz > 1:
        base_U = torch.cat(
            (torch.where(wet_diag[:-1], aV[1:], torch.zeros_like(aV[:-1])),
             torch.zeros_like(aV[-1:])),
            dim=0,
        )
    else:
        base_U = torch.zeros_like(aV)

    diag = [1.0 / base_C[0]]
    upper = [base_U[0] * diag[0]]
    for k in range(1, nz):
        denom = base_C[k] - zML[k] * upper[k - 1]
        diag.append(1.0 / denom)
        upper.append(base_U[k] * diag[k])
    zMC = torch.stack(diag, 0)
    zMU = torch.stack(upper, 0)

    zMC = torch.where(wet_diag, zMC, torch.ones_like(zMC))
    zML = torch.where(wet_diag, zML, torch.zeros_like(zML))
    zMU = torch.where(wet_diag, zMU, torch.zeros_like(zMU))
    return CG3DPreconditioner(zMC=zMC, zML=zML, zMU=zMU)


def _record_fast_thomas(outcome: str, *, reason: str | None = None) -> None:
    """Dispatch record for LOCOMGCM_CG3D_USE_FAST_THOMAS.

    Taken at the fused-kernel call site (hit / error) or at the point where the
    residual is found unusable (inapplicable).  Never at the gate predicate.
    """
    from locomgcm.accel import current_rank, record_dispatch

    record_dispatch(
        "cg3d_fast_thomas",
        outcome=outcome,
        variant="vertical_line_thomas",
        site="cg3d_preconditioner.apply_cg3d_preconditioner",
        backend="cuda",
        rank=current_rank(),
        reason=reason,
    )


def apply_cg3d_preconditioner(
    residual: torch.Tensor,
    preconditioner: CG3DPreconditioner,
) -> torch.Tensor:
    """Apply the 4-pass vertical-line preconditioner to ``residual``."""
    if residual.ndim != 4:
        raise ValueError(
            f"residual must be [batch,nz,ny,nx], got {tuple(residual.shape)}"
        )
    if preconditioner.zMC.shape != residual.shape[1:]:
        raise ValueError("preconditioner shape must match residual vertical grid")

    # Capability declines may use the reference before selection. Once the
    # native implementation is called, errors always propagate. The legacy
    # LOCOMGCM_CG3D_FAST_THOMAS_STRICT flag no longer changes this failure contract.
    from locomgcm.accel import gate

    fast_thomas_requested = gate("cg3d_fast_thomas").effective_value
    # The native apply is a custom operator with a backward only: a live
    # forward mode would get its output back without the tangent and without an
    # error.  Inside the ROOT path's Function forward mode is off.
    forward_mode = forward_mode_active()
    if (
        fast_thomas_requested
        and residual.is_cuda
        and residual.dtype == torch.float64
        and not forward_mode
    ):
        try:
            from ._cg3d_thomas import apply_cg3d_preconditioner_fast

            fast_q = apply_cg3d_preconditioner_fast(residual, preconditioner)
        except Exception:
            _record_fast_thomas("error", reason="fused_thomas_raised")
            raise
        else:
            # The fused CUDA Thomas kernel really ran; this is the ignition
            # line, not the gate predicate above (that one only says requested).
            _record_fast_thomas("hit")
            return fast_q
    elif fast_thomas_requested:
        # Requested but this residual cannot take the fused kernel.  Declared
        # with a reason instead of a silent eager fallback (capability gate with
        # unmet_policy='fallback', so no raise -- the fallback policy).
        _record_fast_thomas(
            "inapplicable",
            reason=("non_cuda" if not residual.is_cuda
                    else "dtype_not_float64" if residual.dtype != torch.float64
                    else "autograd"),
        )

    # Both sweeps are out of place for the same reason as the factorization
    # above: the level written at step k is read at step k+1, so writing into a
    # preallocated ``q`` invalidates it for autograd.  Arithmetic is unchanged.
    nz = residual.shape[1]
    down = [preconditioner.zMC[0] * residual[:, 0]]
    for k in range(1, nz):
        down.append(
            preconditioner.zMC[k] * (
                residual[:, k] - preconditioner.zML[k] * down[k - 1]
            )
        )
    up = list(down)
    for k in range(nz - 2, -1, -1):
        up[k] = down[k] - preconditioner.zMU[k] * up[k + 1]
    return torch.stack(up, dim=1)
