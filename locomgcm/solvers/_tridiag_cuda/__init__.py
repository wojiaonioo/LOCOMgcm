"""Native CUDA backend for the experimental tridiagonal solver."""

from ._fast import (
    SUPPORTED_DTYPES,
    SUPPORTED_NZ,
    compile_tridiag_cuda_extension,
    tridiag_cuda_launch_config,
    tridiag_solve_cuda_direct,
)
from .dispatcher import tridiag_solve

__all__ = [
    "SUPPORTED_DTYPES",
    "SUPPORTED_NZ",
    "compile_tridiag_cuda_extension",
    "tridiag_cuda_launch_config",
    "tridiag_solve",
    "tridiag_solve_cuda_direct",
]
