"""Internal CUDA fast path for the CG3D vertical-line Thomas preconditioner.

This subpackage is an implementation detail of
:mod:`locomgcm.solvers.cg3d_preconditioner`.  It JIT-compiles a fused
Thomas-apply CUDA kernel on first use and exposes a single entry point,
:func:`apply_cg3d_preconditioner_fast`, which is bitwise-equivalent
(``max|d| = 0``) to the Python reference apply.  Callers outside the
preconditioner module should not import from here directly.
"""
from ._fast import apply_cg3d_preconditioner_fast, fast_thomas_available

__all__ = ["apply_cg3d_preconditioner_fast", "fast_thomas_available"]
