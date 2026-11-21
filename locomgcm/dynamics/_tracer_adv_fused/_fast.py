"""Strict lazy loader for the fused tracer-advection directional pass.

The direct binding accepts eleven four-dimensional float64 or float32 CUDA
tensors (one dtype for all eleven) whose per-dimension extents either match the
pass's iteration space or are 1 (broadcast).  It never inserts contiguous
copies, never falls back, and has no autograd implementation.  Gate selection,
the halo exchanges, and the eager fallback live one level up, in
``dispatcher.py``, which shares this module's :func:`direct_decline_reason`
predicate so the two can never disagree about what the kernel supports.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from locomgcm.accel.tensor_context import extent_mismatch, grad_mode_requires_reference


_HERE = Path(__file__).resolve().parent
_KEY = "tracer_adv_fused_cuda"
_EXTENSION_NAME = "locomgcm_tracer_adv_fused"
#: The dtypes kernel.cu instantiates (AT_DISPATCH_FLOATING_TYPES).
_SUPPORTED_DTYPES = (torch.float64, torch.float32)
_DTYPE_NAMES = {torch.float64: "float64", torch.float32: "float32"}
#: ``(device, dtype name, axis)`` -> the native launch configuration.  The axis
#: is part of the identity: kernel.cu templates on it, so the two axes are two
#: instantiations with different register footprints.
_LAUNCH_CONFIG_CACHE: dict[tuple[int, str, str], dict[str, Any]] = {}
_EXTENSION: Any | None = None
_LOAD_FAILED = False
# Precise division and separate multiply/add rounding are numerical
# requirements: the DST3-FL limiter chain is bitwise only when each eager
# multiply and following add stays two rounded operations.
_CUDA_FLAGS = [
    "-O3",
    "-prec-div=true",
    "-prec-sqrt=true",
    "-fmad=false",
]
_CPP_FLAGS = ["-O3"]

#: Positional contract of the tensor tuple, mirroring ``tracer_adv_pass_fwd``.
TENSOR_NAMES = (
    "t_wide",
    "t_left",
    "t_right",
    "mask_wide",
    "vel_face",
    "vel_trans",
    "recip_dc",
    "recip_vol",
    "mask_in_c",
    "tracer_orig",
    "local_t",
)
_TRACER_ORIG_INDEX = TENSOR_NAMES.index("tracer_orig")
#: Smallest tile extent the DST3 stencil can be evaluated on (``2*halo`` of its
#: ``tracer_flux`` declaration); this mirrors the native TORCH_CHECK.
_MIN_TILE_EXTENT = 4


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
    """Compile once and return the extension; preserve the first build error."""
    global _EXTENSION, _LOAD_FAILED
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_FAILED:
        raise RuntimeError(
            "fused tracer-advection extension previously failed to load"
        )
    from torch.utils.cpp_extension import load

    try:
        _EXTENSION = load(
            name=_EXTENSION_NAME,
            sources=[str(_HERE / "extension.cpp"), str(_HERE / "kernel.cu")],
            extra_cflags=_CPP_FLAGS,
            extra_cuda_cflags=_cuda_flags(),
            with_cuda=True,
            verbose=os.environ.get("LOCOMGCM_TRACER_ADV_FUSED_VERBOSE", "0") == "1",
        )
    except Exception:
        _LOAD_FAILED = True
        raise
    _push_launch_decision()
    return _EXTENSION


def _dtype_name(dtype: torch.dtype) -> str:
    name = _DTYPE_NAMES.get(dtype)
    if name is None:
        raise RuntimeError(
            f"fused tracer advection supports float64 or float32; got {dtype}"
        )
    return name


def _ledger_key(dtype: torch.dtype, axis: str) -> str:
    """Keep the float64 ledger key readable; suffix dtype and axis."""
    suffix = "" if dtype == torch.float64 else f"@{_dtype_name(dtype)}"
    return f"{_KEY}{suffix}@{axis}"


def _get_launch_config(
    device: int, dtype: torch.dtype, axis: str, extension: Any | None = None
) -> dict[str, Any]:
    """Derive once per ``(device, dtype, axis)``; later calls hit this dict."""
    name = _dtype_name(dtype)
    key = (device, name, axis)
    cached = _LAUNCH_CONFIG_CACHE.get(key)
    if cached is not None:
        return cached
    native = _load_extension() if extension is None else extension
    with torch.cuda.device(device):
        config = dict(native.launch_config(device, name, axis))
    _LAUNCH_CONFIG_CACHE[key] = config
    from locomgcm.accel import record_launch_decision

    record_launch_decision(
        kernel=_ledger_key(dtype, axis),
        device=str(torch.device("cuda", device)),
        decision=config,
    )
    return config


def _push_launch_decision() -> None:
    """Mark the load event; derive no instantiation's launch configuration.

    The suffixed rows are reserved for configurations the direct path resolved
    for an actual launch, so the ignition audit never reads a resource basis
    the run did not use.  The ``@load`` marker keeps a loaded package visible
    to the missing-decision audit.
    """
    if not torch.cuda.is_available():
        return
    from locomgcm.accel import record_launch_decision

    device = torch.cuda.current_device()
    record_launch_decision(
        kernel=f"{_KEY}@load",
        device=str(torch.device("cuda", device)),
        decision={"phase": "loaded"},
    )


def compile_tracer_adv_fused_extension() -> Any:
    """Compile/load the extension now, raising the complete build failure."""
    return _load_extension()


def tracer_adv_fused_launch_count() -> int:
    """Return the native extension's successful launch-request count."""
    return int(_load_extension().native_launch_count())


def reset_tracer_adv_fused_launch_count() -> None:
    """Reset the native extension launch-request counter."""
    _load_extension().reset_native_launch_count()


#: One reason token per applicability rule.  The strict entry point raises on
#: these; the production dispatcher records them as ``inapplicable`` and falls
#: back.  Both read the SAME predicate, so the gate can never accept an input
#: the kernel would reject (or decline one it would have served).
_DECLINE_MESSAGES = {
    "arity": "native fused tracer advection requires exactly 11 tensors",
    "non_tensor": "native fused tracer advection requires tensor inputs",
    "axis": "native fused tracer advection axis must be 'x' or 'y'",
    "with_limiter": "native fused tracer advection with_limiter must be bool",
    "emit_tendency": "native fused tracer advection emit_tendency must be bool",
    "has_mask_in_c": "native fused tracer advection has_mask_in_c must be bool",
    "layout": "native fused tracer advection inputs must use strided layout",
    "non_cuda": "native fused tracer advection requires CUDA inputs",
    "device_mismatch": (
        "native fused tracer advection inputs must be on one CUDA device"
    ),
    "dtype": (
        "native fused tracer advection requires float64 or float32 inputs, all "
        "of one dtype"
    ),
    "ndim": "native fused tracer advection inputs must be four-dimensional",
    "empty_shape": "native fused tracer advection inputs must be non-empty",
    "tile_extent": (
        "native fused tracer advection needs at least four cells per axis"
    ),
    "shape": (
        "native fused tracer advection input extents must match the pass "
        "iteration space or be 1"
    ),
    "negative_stride": (
        "native fused tracer advection inputs must not have negative strides"
    ),
    "splice_bounds": (
        "native fused tracer advection splice bounds must lie within the face "
        "count"
    ),
    "autograd": "native fused tracer advection has no autograd implementation",
}


def _expected_shapes(
    axis: str, batch: int, levels: int, ny: int, nx: int
) -> dict[str, tuple[int, int, int, int]]:
    """Iteration-space extents each input is checked against (1 also allowed).

    The offsets are the eager halo extents: ``_X/_Y_WIDE_EXTENT`` is two cells
    on both sides of the tracer, ``_X/_Y_MASK_EXTENT`` pads the ``n`` unique
    faces of the face mask with one ghost before and two after (faces
    ``-1 .. n+1``, length ``n+3``; see ``tracer_flux._dst3_face_mask_halo``), and
    the face fields carry one extra face along the pass axis.
    """
    if axis == "y":
        face = (batch, levels, ny + 1, nx)
        wide = (batch, levels, ny + 4, nx)
        mask = (batch, levels, ny + 3, nx)
    else:
        face = (batch, levels, ny, nx + 1)
        wide = (batch, levels, ny, nx + 4)
        mask = (batch, levels, ny, nx + 3)
    center = (batch, levels, ny, nx)
    return {
        "t_wide": wide,
        "t_left": face,
        "t_right": face,
        "mask_wide": mask,
        "vel_face": face,
        "vel_trans": face,
        "recip_dc": face,
        "recip_vol": center,
        "mask_in_c": center,
        "tracer_orig": center,
        "local_t": center,
    }


def direct_decline_reason(
    values: tuple[torch.Tensor, ...],
    *,
    axis: str,
    with_limiter: bool,
    emit_tendency: bool,
    has_mask_in_c: bool,
    dst_start: int,
    dst_stop: int,
) -> str | None:
    """Return the first unmet applicability rule, or ``None`` when supported."""
    if len(values) != len(TENSOR_NAMES):
        return "arity"
    if any(not isinstance(value, torch.Tensor) for value in values):
        return "non_tensor"
    if axis not in ("x", "y"):
        return "axis"
    if not isinstance(with_limiter, bool):
        return "with_limiter"
    if not isinstance(emit_tendency, bool):
        return "emit_tendency"
    if not isinstance(has_mask_in_c, bool):
        return "has_mask_in_c"
    if any(value.layout != torch.strided for value in values):
        return "layout"
    if any(value.device.type != "cuda" for value in values):
        return "non_cuda"
    reference = values[_TRACER_ORIG_INDEX]
    if any(value.device != reference.device for value in values):
        return "device_mismatch"
    if reference.dtype not in _SUPPORTED_DTYPES:
        return "dtype"
    if any(value.dtype != reference.dtype for value in values):
        return "dtype"
    if any(value.ndim != 4 for value in values):
        return "ndim"
    if any(value.numel() == 0 for value in values):
        return "empty_shape"
    batch, levels, ny, nx = (int(size) for size in reference.shape)
    if ny < _MIN_TILE_EXTENT or nx < _MIN_TILE_EXTENT:
        return "tile_extent"
    expected = _expected_shapes(axis, batch, levels, ny, nx)
    for name, value in zip(TENSOR_NAMES, values):
        if name == "mask_in_c" and not has_mask_in_c:
            # The placeholder carries no data; only its rank and dtype matter.
            continue
        for size, target in zip(value.shape, expected[name]):
            if extent_mismatch(int(size), target):
                return "shape"
    if any(stride < 0 for value in values for stride in value.stride()):
        return "negative_stride"
    face_count = (ny + 1) if axis == "y" else (nx + 1)
    if not 0 <= dst_start <= face_count or not 0 <= dst_stop <= face_count:
        return "splice_bounds"
    if grad_mode_requires_reference():
        return "autograd"
    return None


def _check_direct(
    values: tuple[torch.Tensor, ...],
    *,
    axis: str,
    with_limiter: bool,
    emit_tendency: bool,
    has_mask_in_c: bool,
    dst_start: int,
    dst_stop: int,
) -> None:
    reason = direct_decline_reason(
        values,
        axis=axis,
        with_limiter=with_limiter,
        emit_tendency=emit_tendency,
        has_mask_in_c=has_mask_in_c,
        dst_start=dst_start,
        dst_stop=dst_stop,
    )
    if reason is not None:
        raise RuntimeError(_DECLINE_MESSAGES[reason])


def tracer_adv_pass_cuda_direct(
    values: tuple[torch.Tensor, ...],
    *,
    dt: float,
    dst_start: int,
    dst_stop: int,
    with_limiter: bool,
    has_mask_in_c: bool,
    emit_tendency: bool,
    axis: str,
) -> torch.Tensor:
    """Return one fused directional pass from exactly one native CUDA launch."""
    _check_direct(
        values,
        axis=axis,
        with_limiter=with_limiter,
        emit_tendency=emit_tendency,
        has_mask_in_c=has_mask_in_c,
        dst_start=dst_start,
        dst_stop=dst_stop,
    )
    extension = _load_extension()
    reference = values[_TRACER_ORIG_INDEX]
    device = reference.device.index
    _get_launch_config(
        torch.cuda.current_device() if device is None else int(device),
        reference.dtype,
        axis,
        extension,
    )
    return extension.tracer_adv_pass_fwd(
        *values,
        float(dt),
        int(dst_start),
        int(dst_stop),
        bool(with_limiter),
        bool(has_mask_in_c),
        bool(emit_tendency),
        axis,
    )


__all__ = [
    "TENSOR_NAMES",
    "compile_tracer_adv_fused_extension",
    "direct_decline_reason",
    "tracer_adv_fused_launch_count",
    "tracer_adv_pass_cuda_direct",
    "reset_tracer_adv_fused_launch_count",
]
