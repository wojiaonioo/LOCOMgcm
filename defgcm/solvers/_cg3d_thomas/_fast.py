"""Lazy JIT loader + autograd wrapper for the fused CG3D Thomas kernel.

The kernel reproduces the Python reference apply's exact FP operation order
(``-fmad=false`` disables FMA contraction), so the result is bitwise-equal to
:func:`defgcm.solvers.cg3d_preconditioner.apply_cg3d_preconditioner`
(verified ``max|d| = 0`` on real scs-500m-d4 MIT dumps).  FP64 + CUDA only.

The CG3D preconditioner is SPD, so the transpose solve used in ``backward``
is the same forward kernel applied to the incoming gradient.
"""
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

_HERE = Path(__file__).resolve().parent
_EXTENSION: Any | None = None
_LOAD_FAILED = False
# -fmad=false is load-bearing: it forbids FMA contraction so the kernel's
# a*b+c rounding matches PyTorch's per-op rounding -> exact bitwise parity.
_CUDA_FLAGS = ["-O3", "-prec-div=true", "-prec-sqrt=true", "-fmad=false"]
_CPP_FLAGS = ["-O3"]


def _find_conda_cxx() -> str | None:
    prefix = Path(sys.prefix) / "bin"
    for name in ("x86_64-conda-linux-gnu-g++", "g++", "c++"):
        candidate = prefix / name
        if candidate.exists():
            return str(candidate)
        found = shutil.which(name)
        if found:
            return found
    return None


def _cuda_flags() -> list[str]:
    cxx = _find_conda_cxx()
    if cxx is None:
        return list(_CUDA_FLAGS)
    os.environ.setdefault("CXX", cxx)
    os.environ.setdefault("CUDAHOSTCXX", cxx)
    return [*_CUDA_FLAGS, f"-ccbin={cxx}"]


def _load_extension() -> Any:
    """JIT-compile (once) and return the CUDA extension module.

    Raises on failure; the caller (cg3d_preconditioner) owns the Python
    fallback.  ``_LOAD_FAILED`` latches so we do not retry a broken build
    every step.
    """
    global _EXTENSION, _LOAD_FAILED
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_FAILED:
        raise RuntimeError("cg3d fused Thomas extension previously failed to load")
    from torch.utils.cpp_extension import load

    try:
        _EXTENSION = load(
            name="defgcm_cg3d_thomas",
            sources=[str(_HERE / "extension.cpp"), str(_HERE / "kernel.cu")],
            extra_cflags=_CPP_FLAGS,
            extra_cuda_cflags=_cuda_flags(),
            with_cuda=True,
            verbose=os.environ.get("DEFGCM_CG3D_THOMAS_VERBOSE", "0") == "1",
        )
    except Exception:
        _LOAD_FAILED = True
        raise
    return _EXTENSION


def fast_thomas_available() -> bool:
    """True if the fused kernel can be compiled/loaded on this machine."""
    if not torch.cuda.is_available():
        return False
    try:
        _load_extension()
    except Exception:
        return False
    return True


@torch.library.custom_op("defgcm::cg3d_thomas_apply_fwd", mutates_args=())
def cg3d_thomas_apply_fwd(
    zMC: torch.Tensor,
    zML: torch.Tensor,
    zMU: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    return _load_extension().cg3d_thomas_apply_fwd(zMC, zML, zMU, r)


@cg3d_thomas_apply_fwd.register_fake
def _cg3d_thomas_apply_fwd_fake(zMC, zML, zMU, r):
    return torch.empty_like(r)


def _cg3d_thomas_setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:3])


def _cg3d_thomas_backward(ctx, grad_q):
    zMC, zML, zMU = ctx.saved_tensors
    grad_r = None
    if ctx.needs_input_grad[3]:
        # Preserve the native wrapper's first-order, residual-only gradient.
        with torch.no_grad():
            grad_r = cg3d_thomas_apply_fwd(zMC, zML, zMU, grad_q.contiguous())
    return None, None, None, grad_r


cg3d_thomas_apply_fwd.register_autograd(
    _cg3d_thomas_backward, setup_context=_cg3d_thomas_setup_context,
)


def apply_cg3d_preconditioner_fast(
    residual: torch.Tensor,
    preconditioner: Any,
) -> torch.Tensor:
    """Apply the selected CUDA kernel, propagating build and launch errors."""
    with torch.set_grad_enabled(torch.is_grad_enabled() and residual.requires_grad):
        return cg3d_thomas_apply_fwd(
            preconditioner.zMC.contiguous(),
            preconditioner.zML.contiguous(),
            preconditioner.zMU.contiguous(),
            residual.contiguous(),
        )


__all__ = ["apply_cg3d_preconditioner_fast", "fast_thomas_available"]
