"""Strict lazy loader, launch derivation and kernel call for the tridiagonal CUDA kernel.

The direct entry point consumes arbitrary non-negative input strides and does not
insert contiguous copies. It never falls back and never reads a gate: capability
fallback and ignition accounting belong to ``dispatcher.py``. Once a supported
call is selected, build or launch failures are raised rather than hidden.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch


_HERE = Path(__file__).resolve().parent
EXTENSION_NAME = "defgcm_tridiag_cuda"
# Maximum admitted vertical extent, and the default launch-probe size. The
# runtime-sized Thomas recurrence also admits smaller positive extents.
SUPPORTED_NZ = 90
_KEY = "tridiag_cuda"
_STORAGE_OVERRIDE_ENV = "DEFGCM_TRIDIAG_CUDA_CPRIME_STORAGE"
_BLOCK_SIZE_OVERRIDE_ENV = "DEFGCM_TRIDIAG_CUDA_COLUMNS_PER_BLOCK"
_EXTENSION: Any | None = None
_LOAD_FAILED = False
_LAUNCH_CONFIG_CACHE: dict[tuple[int, int, str], dict[str, Any]] = {}
# Single source for the dtype gate: the kernel is instantiated for exactly these
# scalar types (kernel.cu ``tridiag_cuda_launch``), the native entry re-checks
# the same pair, and ``dispatcher.py`` imports this tuple rather than restating
# a dtype list of its own.
SUPPORTED_DTYPES = (torch.float64, torch.float32)
_DTYPE_NAMES = {torch.float64: "float64", torch.float32: "float32"}
# Precise IEEE division/sqrt and separate multiply/add rounding are load-bearing.
_CUDA_FLAGS = [
    "-O3",
    "-prec-div=true",
    "-prec-sqrt=true",
    "-fmad=false",
]
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
    """Compile once and return the extension; never hide a failed build."""
    global _EXTENSION, _LOAD_FAILED
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_FAILED:
        raise RuntimeError("tridiagonal CUDA extension previously failed to load")
    from torch.utils.cpp_extension import load

    try:
        _EXTENSION = load(
            name=EXTENSION_NAME,
            sources=[str(_HERE / "extension.cpp"), str(_HERE / "kernel.cu")],
            extra_cflags=_CPP_FLAGS,
            extra_cuda_cflags=_cuda_flags(),
            with_cuda=True,
            verbose=os.environ.get("DEFGCM_TRIDIAG_CUDA_VERBOSE", "0") == "1",
        )
    except Exception:
        _LOAD_FAILED = True
        raise
    _push_launch_decision()
    return _EXTENSION


def compile_tridiag_cuda_extension() -> Any:
    """Explicitly compile/load and configure the extension for verification."""
    extension = _load_extension()
    if torch.cuda.is_available():
        _get_launch_config(
            torch.cuda.current_device(), SUPPORTED_NZ, torch.float64, extension
        )
    return extension


def _dtype_name(dtype: torch.dtype) -> str:
    name = _DTYPE_NAMES.get(dtype)
    if name is None:
        raise RuntimeError(
            f"tridiagonal CUDA supports float64 or float32; got {dtype}"
        )
    return name


def _ledger_key(dtype: torch.dtype) -> str:
    """Keep the float64 ledger key byte-identical; suffix any other dtype."""
    return _KEY if dtype == torch.float64 else f"{_KEY}@{_dtype_name(dtype)}"


def _launch_overrides() -> tuple[str, int]:
    storage = os.environ.get(_STORAGE_OVERRIDE_ENV, "").strip().lower()
    block_text = os.environ.get(_BLOCK_SIZE_OVERRIDE_ENV, "").strip()
    if not block_text:
        return storage, 0
    try:
        block_size = int(block_text)
    except ValueError as error:
        raise RuntimeError(
            f"{_BLOCK_SIZE_OVERRIDE_ENV} must be a positive integer"
        ) from error
    if block_size <= 0:
        raise RuntimeError(f"{_BLOCK_SIZE_OVERRIDE_ENV} must be a positive integer")
    return storage, block_size


def _record_launch_decision(
    device: int, config: dict[str, Any], dtype: torch.dtype
) -> None:
    """Publish the native occupancy derivation to the unified launch view."""
    from defgcm.accel import record_launch_decision

    record_launch_decision(
        kernel=_ledger_key(dtype),
        device=str(torch.device("cuda", device)),
        decision=config,
    )


def _push_launch_decision() -> None:
    """Mark the load event; derive no dtype's launch configuration.

    The unsuffixed and ``@<dtype>`` rows are reserved for configurations the
    direct path resolved for an actual launch -- the load-time float64
    derivation this replaces misattributed float32-only runs.  The ``@load``
    marker keeps a loaded package visible to the missing-decision audit.
    """
    if not torch.cuda.is_available():
        return
    from defgcm.accel import record_launch_decision

    device = torch.cuda.current_device()
    record_launch_decision(
        kernel=f"{_KEY}@load",
        device=str(torch.device("cuda", device)),
        decision={"phase": "loaded"},
    )


def _get_launch_config(
    device: int,
    nz: int,
    dtype: torch.dtype,
    extension: Any | None = None,
) -> dict[str, Any]:
    """Derive once per ``(device, nz, dtype)``; solves then only hit this dict."""
    name = _dtype_name(dtype)
    key = (device, nz, name)
    cached = _LAUNCH_CONFIG_CACHE.get(key)
    if cached is not None:
        return cached
    storage_override, block_size_override = _launch_overrides()
    native = _load_extension() if extension is None else extension
    with torch.cuda.device(device):
        config = dict(
            native.launch_config(
                device,
                nz,
                storage_override,
                block_size_override,
                name,
            )
        )
    _LAUNCH_CONFIG_CACHE[key] = config
    _record_launch_decision(device, config, dtype)
    return config


def tridiag_cuda_launch_config(
    device: int | None = None,
    nz: int = SUPPORTED_NZ,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """Return the cached native occupancy decision and its resource basis.

    ``dtype`` selects which template instantiation is measured; the two carry
    different register and shared-memory footprints, so a float32 run must read
    the float32 configuration rather than the float64 one.
    """
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("tridiagonal CUDA launch configuration requires CUDA")
        device = torch.cuda.current_device()
    return _get_launch_config(device, nz, dtype)


@torch.library.custom_op("defgcm::tridiag_fwd", mutates_args=())
def tridiag_solve_cuda_direct(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    mitgcm_order: bool = False,
) -> torch.Tensor:
    """Launch the capability-checked native kernel exactly once."""
    extension = _load_extension()
    device = d.device.index
    if device is None:
        device = torch.cuda.current_device()
    config = _get_launch_config(device, int(d.shape[1]), d.dtype, extension)
    output = extension.tridiag_fwd(
        a,
        b,
        c,
        d,
        config["selected_storage_code"],
        config["selected_block_size"],
        mitgcm_order,
    )
    return output


@tridiag_solve_cuda_direct.register_fake
def _tridiag_fwd_fake(a, b, c, d, mitgcm_order=False):
    return d.new_empty(d.shape)


@torch.library.register_vmap(tridiag_solve_cuda_direct)
def _tridiag_fwd_vmap(info, in_dims, a, b, c, d, mitgcm_order=False):
    """Fold independent mapped columns into the existing physical batch axis.

    No vertical or horizontal grid axis is changed. Unmapped operands are
    broadcast over the mapped batch; the native stride-aware kernel reads
    those values without requiring a new numerical implementation.
    """
    batch = info.batch_size

    def expand(tensor, dim):
        if dim is None:
            return tensor.unsqueeze(0).expand(batch, *tensor.shape)
        return tensor.movedim(dim, 0)

    tensors = tuple(expand(t, dim) for t, dim in zip((a, b, c, d), in_dims))
    physical_shape = tensors[3].shape[1:]
    merged = tuple(t.reshape(batch * physical_shape[0], *physical_shape[1:]) for t in tensors)
    result = tridiag_solve_cuda_direct(*merged, mitgcm_order)
    return result.reshape(batch, *physical_shape), 0


__all__ = [
    "EXTENSION_NAME",
    "SUPPORTED_DTYPES",
    "SUPPORTED_NZ",
    "compile_tridiag_cuda_extension",
    "tridiag_cuda_launch_config",
    "tridiag_solve_cuda_direct",
]
