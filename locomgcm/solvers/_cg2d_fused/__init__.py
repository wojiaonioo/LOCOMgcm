"""Fused CUDA kernels for the CG2D preconditioned-conjugate-gradient body.

Forward-only, CUDA, float64/float32, bit-identical to the eager reference in
:func:`locomgcm.solvers.cg2d.pcg_solve`.  ``_fast.py`` holds the strict
``torch.library`` entry points and the shared gate predicate; ``dispatcher.py``
reads the accelerator gate and writes the ignition ledger.
"""

from ._fast import (
    SUPPORTED_DTYPES,
    cg2d_axpy_p,
    cg2d_spmv,
    cg2d_ssor_precond,
    cg2d_update_xr,
    cg2d_update_xrz,
    cuda_launch_config,
    fused_pcg_available,
    load_extension,
)
from .dispatcher import Cg2dFusedIteration, pcg_iteration_kernels

__all__ = [
    "Cg2dFusedIteration",
    "SUPPORTED_DTYPES",
    "cg2d_axpy_p",
    "cg2d_spmv",
    "cg2d_ssor_precond",
    "cg2d_update_xr",
    "cg2d_update_xrz",
    "cuda_launch_config",
    "fused_pcg_available",
    "load_extension",
    "pcg_iteration_kernels",
]
