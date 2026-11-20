"""Native CUDA backend for the experimental KPP boundary-layer depth operator."""

from ._fast import cuda_launch_config, kpp_bldepth_cuda_direct
from .dispatcher import (
    SUPPORTED_NZ,
    SUPPORTED_TABLE_SHAPE,
    kpp_bldepth_cuda,
)

__all__ = [
    "SUPPORTED_NZ",
    "SUPPORTED_TABLE_SHAPE",
    "cuda_launch_config",
    "kpp_bldepth_cuda",
    "kpp_bldepth_cuda_direct",
]
