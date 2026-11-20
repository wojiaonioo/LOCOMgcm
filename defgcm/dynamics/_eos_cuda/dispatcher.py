"""Unified-runtime dispatcher for the fused JMD95/MDJWF density fields.

Enablement comes from the accelerator gate, never from a private environment
read, and every outcome is recorded once against the unified ignition ledger.
"""

from __future__ import annotations

import os

import torch

from defgcm.accel import current_rank, dispatch_scope, gate, record_dispatch
from defgcm.geometry import StructuredCGrid

from ..equation_of_state import JMD95EquationOfState, MDJWFEquationOfState
from ._fast import _PROMOTED_DTYPES, direct_decline_reason, eos_cuda


_KEY = "eos_cuda"
_SITE = "JMD95EquationOfState.density_fields"
_BACKEND = "cuda"
_EAGER_DENSITY = JMD95EquationOfState.density_fields


def _record_variant(self: object) -> str:
    """Label every record, including declines, the way the consumer does."""
    return "mdjwf" if type(self).__name__ == "MDJWFEquationOfState" else "jmd95"


def _record(
    self: object, outcome: str, *, reason: str | None = None, count: int = 1
) -> None:
    record_dispatch(
        _KEY,
        outcome=outcome,
        variant=_record_variant(self),
        site=_SITE,
        backend=_BACKEND,
        rank=current_rank(),
        reason=reason,
        count=count,
    )


def _explicit_variant(self: object) -> str | None:
    if type(self) is JMD95EquationOfState:
        return "jmd95"
    if type(self) is MDJWFEquationOfState:
        return "mdjwf"
    return None


def _decline_reason(self: object, temp: object, salt: object) -> str | None:
    """Dispatcher rules around the kernel's own predicate.

    ``unsupported_eos`` (which model classes map to a kernel variant) and
    ``dump_enabled`` (a dump only the eager path writes) belong to this
    dispatcher.  Every rule about the tensors is
    :func:`._fast.direct_decline_reason`, the predicate the strict custom op
    raises from; ``widen=True`` because float32 is widened below.
    """
    if _explicit_variant(self) is None:
        return "unsupported_eos"
    reason = direct_decline_reason(temp, salt, widen=True)
    if reason is not None:
        return reason
    if os.environ.get("DEFGCM_EOS_DUMP_DIR"):
        return "dump_enabled"
    return None


def _eager(self, temp, salt, grid, prev_total_phi_hyd):
    return _EAGER_DENSITY(self, temp, salt, grid, prev_total_phi_hyd=prev_total_phi_hyd)


def density_fields(
    self,
    temp: torch.Tensor,
    salt: torch.Tensor,
    grid: StructuredCGrid,
    *,
    prev_total_phi_hyd: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the frozen method signature; use CUDA or its exact eager fallback."""
    if not gate(_KEY).effective_value:
        return _eager(self, temp, salt, grid, prev_total_phi_hyd)

    with dispatch_scope(_KEY):
        reason = _decline_reason(self, temp, salt)
        if reason is not None:
            _record(self, "inapplicable", reason=reason)
            if reason == "unsupported_eos":
                raise RuntimeError(
                    "EOS CUDA dispatch implements only exact JMD95EquationOfState "
                    "and MDJWFEquationOfState instances"
                )
            return _eager(self, temp, salt, grid, prev_total_phi_hyd)

        # As ``density_anomaly_from_fields`` resolves it: the lagged potential
        # comes only from the caller.
        pressure = self._pressure_field(
            grid,
            dtype=temp.dtype,
            device=temp.device,
            target=temp,
            prev_total_phi_hyd=prev_total_phi_hyd,
        )
        # The same predicate again, now with the pressure field; temperature
        # and salinity passed above, so only a pressure rule can fire here.
        pressure_reason = direct_decline_reason(temp, salt, pressure, widen=True)
        if pressure_reason is not None:
            _record(self, "inapplicable", reason=pressure_reason)
            return _eager(self, temp, salt, grid, prev_total_phi_hyd)

        variant = _explicit_variant(self)
        assert variant is not None
        # One semantics for the EOS, whatever the run precision: the eager
        # reference evaluates the polynomials in float64 and narrows the ANOMALY
        # on the way out, so the kernel does the same rather than running a
        # genuinely float32 EOS that would disagree with its own fallback by
        # ~1e-4 relative.  The narrowing order matters and is copied from
        # ``equation_of_state.py``: the anomaly is narrowed first and the
        # absolute density is then formed FROM the narrowed anomaly, so this
        # path is bitwise the eager one rather than merely close to it.
        out_dtype = temp.dtype
        widened = out_dtype in _PROMOTED_DTYPES
        try:
            density_anomaly, density = eos_cuda(
                temp.to(torch.float64) if widened else temp,
                salt.to(torch.float64) if widened else salt,
                pressure.to(torch.float64) if widened else pressure,
                eos_variant=variant,
                reference_density=self.reference_density,
            )
        except Exception as error:
            _record(self, "error", reason=type(error).__name__)
            raise
        if widened:
            density_anomaly = density_anomaly.to(out_dtype)
            density = density_anomaly + density_anomaly.new_tensor(
                self.reference_density
            )
        # Distinguish a widened hit from a native float64 one in the ledger:
        # "float32 and float64 are one semantics" is what this whole change
        # claims, and which steps actually took the widened path is its only
        # run-time evidence.  ``record_dispatch`` only *requires* a reason on a
        # non-hit, so carrying one here costs nothing.
        _record(self, "hit", reason="widened_to_float64" if widened else None)
        return density_anomaly, density


__all__ = [
    "density_fields",
]
