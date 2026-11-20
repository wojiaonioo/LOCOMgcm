"""Native CUDA backend and gated dispatcher for the nonlinear EOS density fields."""

from ._fast import cuda_launch_config, eos_cuda, load_eos_cuda_extension
from .dispatcher import density_fields

__all__ = [
    "cuda_launch_config",
    "density_fields",
    "eos_cuda",
    "load_eos_cuda_extension",
]
