"""Fused CUDA operators for the CG3D PCG iteration.

The forward-only operators support float32 and float64. The strict entry
points and gate predicate are in _fast.py; dispatcher.py supplies the halo
adapter. Kernels operate on array layout without selecting a grid family or
rank count. The standard operators retain the eager pointwise update order.
The graph replay additionally combines updates with inner-product reductions,
which changes their summation order."""

from ._fast import (
    SUPPORTED_DTYPES,
    cg3d_axpy_s,
    cg3d_spmv_halo,
    cg3d_thomas_mask,
    cg3d_update_xr,
    cuda_launch_config,
    fused_pcg_available,
    load_extension,
)
from .dispatcher import (
    Cg3dFusedIteration,
    decline_reason,
    make_center_halo,
    pcg_iteration_kernels,
)

__all__ = [
    "Cg3dFusedIteration",
    "SUPPORTED_DTYPES",
    "cg3d_axpy_s",
    "cg3d_spmv_halo",
    "cg3d_thomas_mask",
    "cg3d_update_xr",
    "cuda_launch_config",
    "decline_reason",
    "fused_pcg_available",
    "load_extension",
    "make_center_halo",
    "pcg_iteration_kernels",
]
