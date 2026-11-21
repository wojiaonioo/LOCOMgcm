"""Native CUDA backend and gated dispatcher for the KPP STATEKPP stage."""

from ._fast import (
    cuda_launch_config,
    kpp_statekpp_cuda_direct,
    load_kpp_statekpp_cuda_extension,
)
from .dispatcher import kpp_statekpp_cuda

__all__ = [
    "cuda_launch_config",
    "kpp_statekpp_cuda",
    "kpp_statekpp_cuda_direct",
    "load_kpp_statekpp_cuda_extension",
]
