"""Fused horizontal tracer-advection passes (gate: tracer_adv_fused_cuda)."""

from ._fast import (
    compile_tracer_adv_fused_extension,
    direct_decline_reason,
    tracer_adv_fused_launch_count,
    tracer_adv_pass_cuda_direct,
    reset_tracer_adv_fused_launch_count,
)
from .dispatcher import (
    fused_horizontal_multidim_tendency,
    load_extension,
    tracer_adv_fused_pass,
    tracer_adv_pass_dispatch,
)

__all__ = [
    "compile_tracer_adv_fused_extension",
    "direct_decline_reason",
    "fused_horizontal_multidim_tendency",
    "load_extension",
    "tracer_adv_fused_launch_count",
    "tracer_adv_fused_pass",
    "tracer_adv_pass_cuda_direct",
    "tracer_adv_pass_dispatch",
    "reset_tracer_adv_fused_launch_count",
]
