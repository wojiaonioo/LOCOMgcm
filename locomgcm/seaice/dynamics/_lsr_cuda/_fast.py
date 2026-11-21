"""Lazy loader and direct launches of the sea-ice LSR CUDA extension.

No gate, no fallback: the dispatcher decides applicability and records each
outcome. A failed build or launch raises.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

_HERE = Path(__file__).resolve().parent
EXTENSION_NAME = "locomgcm_seaice_lsr_cuda"
# Separate roundings of every product and sum, IEEE division (kernel.cu header).
_CUDA_FLAGS = ["-O3", "-prec-div=true", "-prec-sqrt=true", "-fmad=false"]
_CPP_FLAGS = ["-O3"]
_EXTENSION: Any | None = None
_LOAD_FAILED = False


def _host_compiler() -> str | None:
    prefix = Path(sys.prefix) / "bin"
    for name in ("x86_64-conda-linux-gnu-g++", "g++", "c++"):
        if (prefix / name).exists():
            return str(prefix / name)
        found = shutil.which(name)
        if found:
            return found
    return None


def load_extension() -> Any:
    """Compile once per process (registry preload target); never hide a failure."""
    global _EXTENSION, _LOAD_FAILED
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_FAILED:
        raise RuntimeError("sea-ice LSR CUDA extension previously failed to load")
    from torch.utils.cpp_extension import load

    flags = list(_CUDA_FLAGS)
    cxx = _host_compiler()
    if cxx is not None:
        os.environ.setdefault("CXX", cxx)
        os.environ.setdefault("CUDAHOSTCXX", cxx)
        flags.append(f"-ccbin={cxx}")
    try:
        _EXTENSION = load(name=EXTENSION_NAME,
                          sources=[str(_HERE / "extension.cpp"), str(_HERE / "kernel.cu")],
                          extra_cflags=_CPP_FLAGS, extra_cuda_cflags=flags, with_cuda=True,
                          verbose=os.environ.get("LOCOMGCM_SEAICE_LSR_CUDA_VERBOSE", "0") == "1")
    except Exception:
        _LOAD_FAILED = True
        raise
    return _EXTENSION


def factor_cuda(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, transpose: bool) -> torch.Tensor:
    return load_extension().factor(a, b, c, transpose)


def sweep_cuda(coef_u, factors_u, old_u, relax_u, coef_v, factors_v, old_v, relax_v,
               swept_u: bool, swept_v: bool) -> tuple[torch.Tensor, torch.Tensor]:
    u, v = load_extension().sweep(coef_u, factors_u, old_u, relax_u, coef_v, factors_v,
                                  old_v, relax_v, swept_u, swept_v)
    return u, v


def ordered_sum_cuda(block: torch.Tensor) -> torch.Tensor:
    return load_extension().ordered_sum(block)


def ordered_sum_pair_cuda(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return load_extension().ordered_sum_pair(first, second)
