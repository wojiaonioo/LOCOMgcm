"""Strict lazy loader for the DST3/DST3-FL transport-flux CUDA core.

The direct binding accepts eleven broadcast-compatible three- or
four-dimensional float64 or float32 CUDA tensors (one dtype for all eleven)
with arbitrary non-negative strides.  It
never inserts contiguous copies or falls back, and supplies a first-order
native VJP for all eleven inputs.  Gate selection and the eager fallback live one level up, in
``dispatcher.py``, which shares this module's :func:`direct_decline_reason`
predicate so the two can never disagree about what the kernel supports.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Any, List

import torch
from defgcm.accel.tensor_context import extent_mismatch


_HERE = Path(__file__).resolve().parent
_KEY = "dst3_cuda"
_EXTENSION_NAME = "defgcm_dst3_cuda"
#: The dtypes kernel.cu instantiates (AT_DISPATCH_FLOATING_TYPES).
_SUPPORTED_DTYPES = (torch.float64, torch.float32)
_DTYPE_NAMES = {torch.float64: "float64", torch.float32: "float32"}
#: ``(device, dtype name)`` -> the native launch configuration.  This kernel's
#: occupancy query carries no shape key, so extents are not part of the identity.
_LAUNCH_CONFIG_CACHE: dict[tuple[int, str], dict[str, Any]] = {}
_EXTENSION: Any | None = None
_LOAD_FAILED = False
# Precise division and separate multiply/add rounding are numerical requirements.
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
    """Compile once and return the extension; preserve the first build error."""
    global _EXTENSION, _LOAD_FAILED
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_FAILED:
        raise RuntimeError("DST3 CUDA extension previously failed to load")
    from torch.utils.cpp_extension import load

    try:
        _EXTENSION = load(
            name=_EXTENSION_NAME,
            sources=[str(_HERE / name) for name in ("extension.cpp", "kernel.cu", "backward.cu")],
            extra_cflags=_CPP_FLAGS,
            extra_cuda_cflags=_cuda_flags(),
            with_cuda=True,
            verbose=os.environ.get("DEFGCM_DST3_CUDA_VERBOSE", "0") == "1",
        )
    except Exception:
        _LOAD_FAILED = True
        raise
    _push_launch_decision()
    return _EXTENSION


def _dtype_name(dtype: torch.dtype) -> str:
    name = _DTYPE_NAMES.get(dtype)
    if name is None:
        raise RuntimeError(f"DST3 CUDA supports float64 or float32; got {dtype}")
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
    device: int, dtype: torch.dtype, extension: Any | None = None
) -> dict[str, Any]:
    """Derive once per ``(device, dtype)``; later calls hit this dict.

    kernel.cu dispatches on the input dtype, so a float32 call runs
    ``dst3_transport_flux_kernel<float>`` with its own register footprint.
    Publishing only the extension-load float64 row would leave the ignition
    audit reading a resource basis the run never used.
    """
    name = _dtype_name(dtype)
    key = (device, name)
    cached = _LAUNCH_CONFIG_CACHE.get(key)
    if cached is not None:
        return cached
    native = _load_extension() if extension is None else extension
    with torch.cuda.device(device):
        config = dict(native.launch_config(device, name))
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


def compile_dst3_cuda_extension() -> Any:
    """Compile/load the extension now, raising the complete build failure."""
    return _load_extension()


def dst3_cuda_launch_config(
    device: int | None = None, dtype: torch.dtype = torch.float64
) -> dict[str, Any]:
    """Return the native occupancy decision cached for this kernel and device.

    ``dtype`` selects which template instantiation is measured; the two carry
    different register footprints, so a float32 run must read the float32
    configuration rather than the float64 one.
    """
    _dtype_name(dtype)
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("DST3 CUDA launch configuration requires CUDA")
        device = torch.cuda.current_device()
    return _get_launch_config(device, dtype)


def dst3_cuda_launch_count() -> int:
    """Return the native extension's successful launch-request count."""
    return int(_load_extension().native_launch_count())


def reset_dst3_cuda_launch_count() -> None:
    """Reset the native extension launch-request counter."""
    _load_extension().reset_native_launch_count()


#: One reason token per applicability rule.  The strict entry point raises on
#: these; the production dispatcher records them as ``inapplicable`` and falls
#: back.  Both read the SAME predicate, so the gate can never accept an input
#: the kernel would reject (or decline one it would have served).
_DECLINE_MESSAGES = {
    "arity": "native DST3 CUDA dispatch requires exactly 11 tensors",
    "non_tensor": "native DST3 CUDA dispatch requires tensor inputs",
    "layout": "native DST3 CUDA inputs must use strided layout",
    "non_cuda": "native DST3 CUDA dispatch requires CUDA inputs",
    "device_mismatch": "native DST3 CUDA inputs must be on one CUDA device",
    "dtype": (
        "native DST3 CUDA dispatch requires float64 or float32 inputs, all of "
        "one dtype"
    ),
    "ndim": "native DST3 CUDA inputs must be three- or four-dimensional",
    "trailing_shape": "native DST3 CUDA trailing input dimensions must match",
    "empty_shape": "native DST3 CUDA inputs must have positive extents",
    "batch_broadcast": (
        "native DST3 CUDA leading batch dimensions are not broadcast-compatible"
    ),
    "negative_stride": "native DST3 CUDA inputs must not have negative strides",
    "with_limiter": "native DST3 CUDA with_limiter must be bool",
    "torch_transform": "native DST3 supports first-order reverse only",
}


def direct_decline_reason(
    values: tuple[torch.Tensor, ...], *, with_limiter: bool
) -> str | None:
    """Return the first unmet applicability rule, or ``None`` when supported."""
    if len(values) != 11:
        return "arity"
    if any(not isinstance(value, torch.Tensor) for value in values):
        return "non_tensor"
    reference = values[0]
    if any(value.layout != torch.strided for value in values):
        return "layout"
    if any(value.device.type != "cuda" for value in values):
        return "non_cuda"
    if any(value.device != reference.device for value in values):
        return "device_mismatch"
    # Mirrors AT_DISPATCH_FLOATING_TYPES in kernel.cu: one instantiation per
    # floating dtype, and every input feeds the same instantiation.
    if reference.dtype not in _SUPPORTED_DTYPES:
        return "dtype"
    if any(value.dtype != reference.dtype for value in values):
        return "dtype"
    if any(value.ndim not in (3, 4) for value in values):
        return "ndim"
    trailing_shape = reference.shape[-3:]
    if any(value.shape[-3:] != trailing_shape for value in values[1:]):
        return "trailing_shape"
    if any(value.numel() == 0 for value in values):
        return "empty_shape"
    batch_extents = tuple(value.shape[0] if value.ndim == 4 else 1 for value in values)
    output_batch = max(batch_extents)
    if any(extent_mismatch(batch, output_batch) for batch in batch_extents):
        return "batch_broadcast"
    if any(stride < 0 for value in values for stride in value.stride()):
        return "negative_stride"
    if not isinstance(with_limiter, bool):
        return "with_limiter"
    if torch._C._are_functorch_transforms_active() or any(
        torch.autograd.forward_ad.unpack_dual(value).tangent is not None
        for value in values
    ):
        return "torch_transform"
    return None


def _check_direct(values: tuple[torch.Tensor, ...], *, with_limiter: bool) -> None:
    reason = direct_decline_reason(values, with_limiter=with_limiter)
    if reason is not None:
        raise RuntimeError(_DECLINE_MESSAGES[reason])


def dst3_transport_flux_values_cuda_direct(
    T_left: torch.Tensor,
    T_right: torch.Tensor,
    Tm2: torch.Tensor,
    Tm1: torch.Tensor,
    T0: torch.Tensor,
    Tp1: torch.Tensor,
    mask_im1: torch.Tensor,
    mask_i: torch.Tensor,
    mask_ip1: torch.Tensor,
    u_trans: torch.Tensor,
    u_cfl: torch.Tensor,
    *,
    with_limiter: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return both transport fluxes from exactly one native CUDA launch."""
    values = (
        T_left,
        T_right,
        Tm2,
        Tm1,
        T0,
        Tp1,
        mask_im1,
        mask_i,
        mask_ip1,
        u_trans,
        u_cfl,
    )
    _check_direct(values, with_limiter=with_limiter)
    return dst3_transport_flux_fwd(list(values), with_limiter)


@torch.library.custom_op("defgcm::dst3_transport_flux_fwd", mutates_args=())
def dst3_transport_flux_fwd(
    values: List[torch.Tensor],
    with_limiter: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    extension = _load_extension()
    # Publish the configuration this launch actually runs under; kernel.cu
    # dispatches on T_left's dtype.  Cached after the first call per (device,
    # dtype).
    T_left = values[0]
    device = T_left.device.index
    _get_launch_config(
        torch.cuda.current_device() if device is None else int(device),
        T_left.dtype,
        extension,
    )
    upwind_flux, flux_dst = extension.dst3_transport_flux_fwd(*values, with_limiter)
    _record_execution("forward", with_limiter)
    return upwind_flux, flux_dst


@dst3_transport_flux_fwd.register_fake
def _dst3_transport_flux_fwd_fake(values, with_limiter):
    shape = torch.broadcast_shapes(*(value.shape for value in values))
    return values[0].new_empty(shape), values[0].new_empty(shape)



def _record_execution(direction, limited, outcome="hit", reason=None):
    from defgcm.accel import current_rank, gate, record_dispatch
    if gate(_KEY).effective_value:
        record_dispatch(
            _KEY, outcome=outcome, variant=direction,
            reason=reason, site="dst3_" + ("limiter" if limited else "plain"),
            backend="cuda", rank=current_rank(),
        )


@torch.library.custom_op(
    "defgcm::dst3_transport_flux_bwd", mutates_args=(),
    schema="(Tensor[] values, Tensor?[] seeds, bool limited, bool[] needs) -> Tensor?[]",
)
def dst3_transport_flux_bwd(values, seeds, limited, needs):
    try:
        extension = _load_extension()
        device = values[0].device.index
        key = (device, values[0].dtype)
        if key not in _BACKWARD_CONFIGS:
            from defgcm.accel import record_launch_decision
            config = dict(extension.backward_launch_config(device, _dtype_name(values[0].dtype)))
            record_launch_decision(kernel=_ledger_key(values[0].dtype) + "@backward",
                                   device=str(values[0].device), decision=config)
            _BACKWARD_CONFIGS.add(key)
        output = list(extension.dst3_transport_flux_bwd(values, seeds, limited, needs))
    except Exception as error:
        _record_execution("backward", limited, "error", type(error).__name__)
        raise
    _record_execution("backward", limited)
    return output


_BACKWARD_CONFIGS = set()


@dst3_transport_flux_bwd.register_fake
def _dst3_transport_flux_bwd_fake(values, seeds, limited, needs):
    return [v.new_empty(v.shape) if need else None for v, need in zip(values, needs)]


def _setup_context(ctx, inputs, output):
    values, ctx.limited = inputs
    ctx.save_for_backward(*values)
    ctx.set_materialize_grads(False)


def _backward(ctx, upwind_seed, dst_seed):
    if torch.is_grad_enabled():
        raise RuntimeError("native DST3 supports first-order reverse only; "
                           "disable dst3_cuda for higher-order derivatives")
    needs = list(ctx.needs_input_grad[0])
    if upwind_seed is None:
        needs[0] = needs[1] = False
    if dst_seed is None:
        for index in (*range(2, 9), 10):
            needs[index] = False
    gradients = dst3_transport_flux_bwd(
        list(ctx.saved_tensors), [upwind_seed, dst_seed],
        ctx.limited, needs,
    )
    return gradients, None


dst3_transport_flux_fwd.register_autograd(_backward, setup_context=_setup_context)


__all__ = [
    "compile_dst3_cuda_extension",
    "direct_decline_reason",
    "dst3_cuda_launch_config",
    "dst3_cuda_launch_count",
    "dst3_transport_flux_values_cuda_direct",
    "reset_dst3_cuda_launch_count",
]
