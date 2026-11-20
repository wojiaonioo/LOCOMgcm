"""Strict lazy loader for the implicit-diffusion assembly CUDA extension.

The low-level entry point accepts arbitrary non-negative input strides and never
inserts a contiguous copy. It raises on inputs outside its narrow CUDA contract;
the public dispatcher owns reference fallbacks and ignition accounting.  Both
decide applicability with one predicate, :func:`direct_decline_reason`.
"""

import os
import shutil
import sys
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from defgcm.accel.tensor_context import extent_mismatch, grad_mode_requires_reference


_HERE = Path(__file__).resolve().parent
_KEY = "tridiag_build_cuda"
SUPPORTED_NZ = 90
# The kernel is instantiated for exactly these scalar types (kernel.cu
# ``launch_typed``/``tridiag_build_cuda_launch``); :func:`direct_decline_reason`
# is the only place that reads them.
SUPPORTED_DTYPES = (torch.float64, torch.float32)
_DTYPE_NAMES = {torch.float64: "float64", torch.float32: "float32"}
#: ``(device, nz, dtype name)`` -> the native launch configuration.
_LAUNCH_CONFIG_CACHE: dict[tuple[int, int, str], dict[str, Any]] = {}
_EXTENSION: Any | None = None
_LOAD_FAILED = False
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


def cuda_compile_flags() -> list[str]:
    """Return the exact CUDA flags supplied to ``cpp_extension.load``."""
    cxx = _find_conda_cxx()
    if cxx is None:
        return list(_CUDA_FLAGS)
    os.environ.setdefault("CXX", cxx)
    os.environ.setdefault("CUDAHOSTCXX", cxx)
    return [*_CUDA_FLAGS, f"-ccbin={cxx}"]


def load_extension() -> Any:
    """Compile on first use and return the native extension module."""
    global _EXTENSION, _LOAD_FAILED
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_FAILED:
        raise RuntimeError("tridiag-build CUDA extension previously failed to load")
    from torch.utils.cpp_extension import load

    try:
        _EXTENSION = load(
            name="defgcm_tridiag_build_cuda",
            sources=[str(_HERE / "extension.cpp"), str(_HERE / "kernel.cu")],
            extra_cflags=_CPP_FLAGS,
            extra_cuda_cflags=cuda_compile_flags(),
            with_cuda=True,
            verbose=os.environ.get("DEFGCM_TRIDIAG_BUILD_CUDA_VERBOSE", "0") == "1",
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
            "tridiag-build CUDA supports float64 or float32; got "
            f"{dtype}"
        )
    return name


def _ledger_key(dtype: torch.dtype) -> str:
    """Keep the float64 ledger key byte-identical; suffix any other dtype."""
    return _KEY if dtype == torch.float64 else f"{_KEY}@{_dtype_name(dtype)}"


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


def _get_launch_config(
    device: int,
    nz: int,
    dtype: torch.dtype,
    extension: Any | None = None,
) -> dict[str, Any]:
    """Derive once per ``(device, nz, dtype)``; later builds hit this dict.

    The publication is not optional bookkeeping: kernel.cu dispatches on the
    input dtype, so a float32 assembly runs ``tridiag_build_kernel<float>``'s
    configuration.  Recording only the extension-load float64 row would leave
    the ignition audit reading a resource basis the run never used.
    """
    name = _dtype_name(dtype)
    key = (device, nz, name)
    cached = _LAUNCH_CONFIG_CACHE.get(key)
    if cached is not None:
        return cached
    native = load_extension() if extension is None else extension
    with torch.cuda.device(device):
        config = dict(native.launch_config(device, nz, name))
    _LAUNCH_CONFIG_CACHE[key] = config
    _record_launch_decision(device, config, dtype)
    return config


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


def cuda_launch_config(
    device: int | None = None,
    nz: int = SUPPORTED_NZ,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """Return the native occupancy decision cached for ``(kernel, device, nz)``.

    ``dtype`` selects which template instantiation is measured; the two carry
    different register and shared-memory footprints, so a float32 run must read
    the float32 configuration rather than the float64 one.
    """
    _dtype_name(dtype)
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("tridiag-build CUDA launch configuration requires CUDA")
        device = torch.cuda.current_device()
    return _get_launch_config(device, nz, dtype)


#: One reason token per applicability rule.  The strict entry raises on these;
#: the production dispatcher records them as ``inapplicable`` and falls back.
#: Both read :func:`direct_decline_reason`, so the gate can never accept an
#: input the kernel would reject, nor decline one it would have served.
_DECLINE_MESSAGES = {
    "scalar_diffusivity": (
        "native tridiag-build dispatch requires a diffusivity tensor field"
    ),
    "missing_recip_hfac": "native tridiag-build dispatch requires recip_hFac",
    "non_tensor": "native tridiag-build dispatch requires tensor inputs",
    "non_cuda": "native tridiag-build dispatch requires CUDA inputs",
    "device_mismatch": "native tridiag-build inputs must be on the tracer device",
    "dtype": (
        "native tridiag-build dispatch requires float64 or float32 inputs, "
        "one dtype across all inputs"
    ),
    "unsupported_shape": "native tridiag-build tracer must have shape [B, 90, ny, nx]",
    "empty_shape": "native tridiag-build tracer extents must be positive",
    "layer_shape": (
        "native tridiag-build layer_thickness must be one-dimensional with 90 levels "
        "and interface_spacing with 89 interfaces"
    ),
    "diffusivity_shape": (
        "native tridiag-build diffusivity must have shape [90, ny, nx] or "
        "[1|B, 90, ny, nx]"
    ),
    "recip_hfac_shape": (
        "native tridiag-build recip_hFac must have shape [90, ny, nx] or "
        "[1|B, 90, ny, nx]"
    ),
    "dt_type": "native tridiag-build dt must be a real number",
    "negative_stride": "native tridiag-build inputs must not have negative strides",
    "autograd": "native tridiag-build dispatch has no autograd implementation",
}


def _field_shape_supported(value: torch.Tensor, tracer: torch.Tensor) -> bool:
    if value.ndim == 3:
        return value.shape == tracer.shape[-3:]
    if value.ndim != 4:
        return False
    return (
        not extent_mismatch(value.shape[0], tracer.shape[0])
        and value.shape[1:] == tracer.shape[1:]
    )


def direct_decline_reason(
    tracer: object,
    layer_thickness: object,
    interface_spacing: object,
    diffusivity: object,
    dt: object,
    recip_hfac: object,
) -> str | None:
    """Return the first unmet applicability rule, or ``None`` when supported.

    Reads the arguments of ``build_implicit_diffusion_system``: a 3-D field is
    one batch, which :func:`tridiag_build_cuda` lifts to the kernel's
    ``[1, 90, ny, nx]``.  Every tensor rule mirrors a check of
    ``extension.cpp`` ``build_fwd``.
    """
    if not isinstance(diffusivity, torch.Tensor):
        return "scalar_diffusivity"
    if recip_hfac is None:
        return "missing_recip_hfac"
    values = (tracer, layer_thickness, interface_spacing, diffusivity, recip_hfac)
    if not all(isinstance(value, torch.Tensor) for value in values):
        return "non_tensor"
    if any(value.device.type != "cuda" for value in values):
        return "non_cuda"
    if any(value.device != tracer.device for value in values):
        return "device_mismatch"
    if tracer.dtype not in SUPPORTED_DTYPES:
        return "dtype"
    if any(value.dtype != tracer.dtype for value in values):
        return "dtype"
    if tracer.ndim != 4 or tracer.shape[-3] != SUPPORTED_NZ:
        return "unsupported_shape"
    if any(size == 0 for size in tracer.shape):
        return "empty_shape"
    if (layer_thickness.ndim != 1 or layer_thickness.shape[0] != SUPPORTED_NZ
            or interface_spacing.ndim != 1 or interface_spacing.shape[0] != SUPPORTED_NZ - 1):
        return "layer_shape"
    if not _field_shape_supported(diffusivity, tracer):
        return "diffusivity_shape"
    if not _field_shape_supported(recip_hfac, tracer):
        return "recip_hfac_shape"
    if not isinstance(dt, Real):
        return "dt_type"
    if any(stride < 0 for value in values for stride in value.stride()):
        return "negative_stride"
    if grad_mode_requires_reference():
        return "autograd"
    return None


def _check_direct(
    tracer: object,
    layer_thickness: object,
    interface_spacing: object,
    diffusivity: object,
    dt: object,
    recip_hfac: object,
) -> None:
    reason = direct_decline_reason(
        tracer, layer_thickness, interface_spacing, diffusivity, dt, recip_hfac
    )
    if reason is not None:
        raise RuntimeError(_DECLINE_MESSAGES[reason])


def tridiag_build_cuda(
    tracer: torch.Tensor,
    layer_thickness: torch.Tensor,
    interface_spacing: torch.Tensor,
    diffusivity: torch.Tensor,
    dt: float,
    recip_hfac: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the capability-checked, single-launch CUDA implementation."""
    _check_direct(tracer, layer_thickness, interface_spacing, diffusivity, dt, recip_hfac)
    # The kernel reads [1|B, 90, ny, nx] fields; a 3-D field is one batch.
    diffusivity_4d = diffusivity.unsqueeze(0) if diffusivity.ndim == 3 else diffusivity
    recip_hfac_4d = recip_hfac.unsqueeze(0) if recip_hfac.ndim == 3 else recip_hfac
    return tridiag_build_fwd(
        tracer, layer_thickness, interface_spacing, diffusivity_4d, float(dt), recip_hfac_4d
    )


@torch.library.custom_op("defgcm::tridiag_build_fwd", mutates_args=())
def tridiag_build_fwd(
    tracer: torch.Tensor,
    layer_thickness: torch.Tensor,
    interface_spacing: torch.Tensor,
    diffusivity: torch.Tensor,
    dt: float,
    recip_hfac: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    extension = load_extension()
    device = tracer.device.index
    # Publish the configuration this launch actually runs under.  Cached after
    # the first build of a given (device, nz, dtype).
    _get_launch_config(
        torch.cuda.current_device() if device is None else int(device),
        int(tracer.shape[1]),
        tracer.dtype,
        extension,
    )
    return tuple(
        extension.build_fwd(
            tracer,
            layer_thickness,
            interface_spacing,
            diffusivity,
            dt,
            recip_hfac,
        )
    )


@tridiag_build_fwd.register_fake
def _tridiag_build_fwd_fake(tracer, layer_thickness, interface_spacing, diffusivity, dt, recip_hfac):
    return tuple(torch.empty_like(tracer) for _ in range(4))


__all__ = [
    "SUPPORTED_DTYPES",
    "SUPPORTED_NZ",
    "cuda_compile_flags",
    "cuda_launch_config",
    "direct_decline_reason",
    "load_extension",
    "tridiag_build_cuda",
]
