"""Native CUDA backend for implicit-diffusion system assembly."""

from ._fast import (
    SUPPORTED_DTYPES,
    cuda_compile_flags,
    cuda_launch_config,
    load_extension,
    tridiag_build_cuda,
)
from .dispatcher import SUPPORTED_NZ, build_implicit_diffusion_system

__all__ = [
    "SUPPORTED_DTYPES",
    "SUPPORTED_NZ",
    "build_implicit_diffusion_system",
    "cuda_compile_flags",
    "cuda_launch_config",
    "load_extension",
    "tridiag_build_cuda",
]
