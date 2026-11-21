"""Native CUDA client for the DST3/DST3-FL numerical core (gate: dst3_cuda)."""

from ._fast import (
    compile_dst3_cuda_extension,
    dst3_cuda_launch_config,
    dst3_cuda_launch_count,
    dst3_transport_flux_values_cuda_direct,
    reset_dst3_cuda_launch_count,
)
from .dispatcher import dst3_transport_flux_values_cuda

__all__ = [
    "compile_dst3_cuda_extension",
    "dst3_cuda_launch_config",
    "dst3_cuda_launch_count",
    "dst3_transport_flux_values_cuda",
    "dst3_transport_flux_values_cuda_direct",
    "reset_dst3_cuda_launch_count",
]
