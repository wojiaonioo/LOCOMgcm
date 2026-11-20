"""Strict lazy loader for the fused nonlinear-EOS CUDA extension.

The direct binding accepts four-dimensional float64 CUDA tensors (one dtype for
all three) and arbitrary non-negative input strides.  Pressure may broadcast to
the temperature shape.  A reduced-precision run reaches the kernel through the
dispatcher, which widens to float64 and narrows the result back (see
``_PROMOTED_DTYPES``); the binding itself never converts anything.
The native path provides a first-order VJP. Public transform requests use the
Torch expression; higher-order reverse differentiation of a native invocation
is explicitly rejected. Input layout is preserved without contiguous copies.
The strict custom op and the dispatcher decide applicability with one
predicate, :func:`direct_decline_reason`.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from defgcm.accel.tensor_context import extent_mismatch

_HERE = Path(__file__).resolve().parent
_KEY = "eos_cuda"
#: The dtypes kernel.cu instantiates.  ``direct_decline_reason`` reads it for
#: both the strict entry and the dispatcher, so the gate cannot drift from the
#: kernel.
_SUPPORTED_DTYPES = (torch.float64,)
#: Dtypes the DISPATCHER serves by widening to float64, launching the float64
#: instantiation and narrowing the result back.  The eager reference already
#: evaluates the JMD95/MDJWF polynomials in float64 for a reduced-precision
#: input (``equation_of_state.py:434-437`` / ``:556-560`` / ``:678-682``), so a
#: genuinely float32 kernel would be a SECOND semantics for the same model: the
#: two answers differ by ~1e-4 relative, and the difference is amplified by the
#: density differences downstream.  Widening here keeps one semantics, and it is
#: bitwise the eager one.  Not part of ``_SUPPORTED_DTYPES``: the direct binding
#: below stays strict, and float32 must never reach the kernel unwidened.
_PROMOTED_DTYPES = (torch.float32,)
_EXTENSION: Any | None = None
_LOAD_FAILED = False
_CUDA_FLAGS = [
    "-O3",
    "-prec-div=true",
    "-prec-sqrt=true",
    "-fmad=false",
]
_CPP_FLAGS = ["-O3"]
_VARIANTS = {"jmd95": 0, "mdjwf": 1}


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
    global _EXTENSION, _LOAD_FAILED
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_FAILED:
        raise RuntimeError("nonlinear-EOS CUDA extension previously failed to load")
    from torch.utils.cpp_extension import load

    try:
        _EXTENSION = load(
            name="defgcm_eos_cuda",
            sources=[str(_HERE / "extension.cpp"), str(_HERE / "kernel.cu")],
            extra_cflags=_CPP_FLAGS,
            extra_cuda_cflags=_cuda_flags(),
            with_cuda=True,
            verbose=os.environ.get("DEFGCM_EOS_CUDA_VERBOSE", "0") == "1",
        )
    except Exception:
        _LOAD_FAILED = True
        raise
    _push_launch_decision()
    return _EXTENSION


def _record_launch_decision(device: int, config: dict[str, Any]) -> None:
    """Publish the native occupancy derivation to the unified launch view."""
    from defgcm.accel import record_launch_decision

    record_launch_decision(
        kernel=_KEY, device=str(torch.device("cuda", device)), decision=config
    )


def _push_launch_decision() -> None:
    """Derive once for the current device so extension load is observable."""
    if not torch.cuda.is_available():
        return
    device = torch.cuda.current_device()
    with torch.cuda.device(device):
        config = dict(_EXTENSION.launch_config(device))
    _record_launch_decision(device, config)


def load_eos_cuda_extension() -> Any:
    """Compile/load the extension now, raising the complete build failure."""
    return _load_extension()


def cuda_launch_config(device: int | None = None) -> dict[str, Any]:
    """Return the native occupancy decision cached for this kernel and device."""
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("nonlinear-EOS CUDA launch configuration requires CUDA")
        device = torch.cuda.current_device()
    with torch.cuda.device(device):
        return dict(_load_extension().launch_config(device))


#: One reason token per applicability rule.  The strict custom op raises on
#: these; the production dispatcher records them as ``inapplicable`` and falls
#: back.  Both read :func:`direct_decline_reason`, so the gate can never accept
#: an input the kernel would reject, nor decline one it would have served.
_DECLINE_MESSAGES = {
    "non_tensor": "native EOS CUDA dispatch requires tensor inputs",
    "non_cuda": "native EOS CUDA dispatch requires CUDA inputs",
    "device_mismatch": "native EOS CUDA inputs must be on one device",
    "dtype": (
        "native EOS CUDA dispatch requires float64 inputs (reduced "
        "precision is widened by the dispatcher, never by the kernel)"
    ),
    "shape": (
        "native EOS CUDA inputs must be four-dimensional and temp and salt "
        "shapes must match"
    ),
    "empty_shape": "native EOS CUDA inputs must have positive extents",
    "negative_stride": "native EOS CUDA inputs must not have negative strides",
    "torch_transform": (
        "native EOS CUDA has no JVP or vmap rule; transforms use the Torch "
        "expression"
    ),
    "pressure_non_tensor": "native EOS CUDA dispatch requires tensor inputs",
    "pressure_device": (
        "native EOS CUDA dispatch requires CUDA inputs on one device; "
        "pressure is not on the temp device"
    ),
    "pressure_layout": (
        "native EOS CUDA pressure must be four-dimensional and share the temp "
        "dtype (float64; reduced precision is widened by the dispatcher, never "
        "by the kernel)"
    ),
    "pressure_shape": "native EOS CUDA pressure must broadcast to temp",
    "pressure_negative_stride": "native EOS CUDA inputs must not have negative strides",
}


#: Default ``pressure`` of ``direct_decline_reason``: judge temperature and
#: salinity alone.  A sentinel rather than ``None``, so a ``None`` pressure is
#: declined as ``pressure_non_tensor`` like any other non-tensor.
_PRESSURE_NOT_BUILT = object()


def direct_decline_reason(
    temp: object,
    salt: object,
    pressure: object = _PRESSURE_NOT_BUILT,
    *,
    widen: bool = False,
) -> str | None:
    """Return the first unmet applicability rule, or ``None`` when supported.

    Omitting ``pressure`` judges temperature and salinity alone: the dispatcher
    asks that before it builds the pressure field, then asks again with it.
    ``widen=True`` is the dispatcher's question: it widens ``_PROMOTED_DTYPES``
    to float64 before the launch, so those dtypes are served as well.  Every
    rule below mirrors a check of ``extension.cpp`` ``eos_fwd``.
    """
    if not isinstance(temp, torch.Tensor) or not isinstance(salt, torch.Tensor):
        return "non_tensor"
    if temp.device.type != "cuda" or salt.device.type != "cuda":
        return "non_cuda"
    if temp.device != salt.device:
        return "device_mismatch"
    dtypes = _SUPPORTED_DTYPES + _PROMOTED_DTYPES if widen else _SUPPORTED_DTYPES
    if temp.dtype not in dtypes or salt.dtype != temp.dtype:
        return "dtype"
    if temp.ndim != 4 or salt.ndim != 4 or temp.shape != salt.shape:
        return "shape"
    if temp.numel() == 0:
        return "empty_shape"
    if any(stride < 0 for value in (temp, salt) for stride in value.stride()):
        return "negative_stride"
    if requires_torch_transform(temp, salt):
        return "torch_transform"
    if pressure is _PRESSURE_NOT_BUILT:
        return None
    if not isinstance(pressure, torch.Tensor):
        return "pressure_non_tensor"
    if pressure.device.type != "cuda" or pressure.device != temp.device:
        return "pressure_device"
    if pressure.dtype != temp.dtype or pressure.ndim != 4:
        return "pressure_layout"
    if any(
        extent_mismatch(actual, expected)
        for actual, expected in zip(pressure.shape, temp.shape)
    ):
        return "pressure_shape"
    if any(stride < 0 for stride in pressure.stride()):
        return "pressure_negative_stride"
    if requires_torch_transform(pressure):
        return "torch_transform"
    return None


def _check_direct(values: tuple[torch.Tensor, ...]) -> None:
    temp, salt, pressure = values
    reason = direct_decline_reason(temp, salt, pressure)
    if reason is not None:
        raise RuntimeError(_DECLINE_MESSAGES[reason])


@torch.library.custom_op("defgcm::eos_fwd", mutates_args=())
def eos_fwd(
    temp: torch.Tensor,
    salt: torch.Tensor,
    pressure: torch.Tensor,
    variant: str,
    reference_density: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one strict EOS launch behind an opaque dispatcher boundary."""
    values = (temp, salt, pressure)
    _check_direct(values)
    try:
        variant_id = _VARIANTS[variant]
    except KeyError as error:
        raise RuntimeError(
            f"native EOS CUDA variant must be one of {tuple(_VARIANTS)}, got {variant!r}"
        ) from error
    density_anomaly, density = _load_extension().eos_fwd(
        temp,
        salt,
        pressure,
        variant_id,
        float(reference_density),
    )
    return density_anomaly, density


@eos_fwd.register_fake
def _eos_fwd_fake(
    temp, salt, pressure, variant, reference_density
) -> tuple[torch.Tensor, torch.Tensor]:
    del salt, pressure, variant, reference_density
    return tuple(torch.empty(temp.shape, device=temp.device, dtype=temp.dtype)
                 for _ in range(2))


@torch.library.custom_op(
    "defgcm::eos_bwd", mutates_args=(),
    schema="(Tensor temp, Tensor salt, Tensor pressure, Tensor density_seed, "
           "Tensor? alpha_seed, Tensor? beta_seed, str variant, float reference_density, "
           "bool need_temp=True, bool need_salt=True, bool need_pressure=True) "
           "-> (Tensor?, Tensor?, Tensor?)",
)
def eos_bwd(
    temp: torch.Tensor,
    salt: torch.Tensor,
    pressure: torch.Tensor,
    density_seed: torch.Tensor,
    alpha_seed: torch.Tensor | None,
    beta_seed: torch.Tensor | None,
    variant: str,
    reference_density: float,
    need_temp: bool = True,
    need_salt: bool = True,
    need_pressure: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Pointwise T/S/Pa VJP; broadcast reduction belongs to the caller."""
    return tuple(_load_extension().eos_bwd(
        temp, salt, pressure, density_seed, alpha_seed, beta_seed,
        _VARIANTS[variant], reference_density, need_temp, need_salt, need_pressure,
    ))


@eos_bwd.register_fake
def _eos_bwd_fake(temp, salt, pressure, density_seed, alpha_seed, beta_seed,
                  variant, reference_density, need_temp=True, need_salt=True,
                  need_pressure=True):
    return tuple(torch.empty(temp.shape, device=temp.device, dtype=temp.dtype)
                 if needed else None
                 for needed in (need_temp, need_salt, need_pressure))


def _eos_setup_context(ctx, inputs, output):
    temp, salt, pressure, variant, reference_density = inputs
    ctx.save_for_backward(temp, salt, pressure)
    ctx.variant = variant
    ctx.reference_density = reference_density
    ctx.set_materialize_grads(False)


def _eos_backward(ctx, anomaly_seed, density_seed):
    if torch.is_grad_enabled():
        raise RuntimeError(
            "native EOS supports first-order reverse differentiation only; "
            "use the Torch EOS expression for higher-order derivatives"
        )
    temp, salt, pressure = ctx.saved_tensors
    if anomaly_seed is None:
        seed = density_seed
    elif density_seed is None:
        seed = anomaly_seed
    else:
        seed = anomaly_seed + density_seed
    if seed is None:
        return None, None, None, None, None
    gt, gs, gp = eos_bwd(temp, salt, pressure, seed, None, None,
                        ctx.variant, ctx.reference_density,
                        *ctx.needs_input_grad[:3])
    return gt, gs, gp.sum_to_size(pressure.shape) if gp is not None else None, None, None


eos_fwd.register_autograd(_eos_backward, setup_context=_eos_setup_context)


def requires_torch_transform(*values: torch.Tensor) -> bool:
    """Opaque native EOS has no JVP/vmap rule; preserve the public transforms."""
    return torch._C._are_functorch_transforms_active() or any(
        torch.autograd.forward_ad.unpack_dual(value).tangent is not None
        for value in values
    )


def eos_cuda(
    temp: torch.Tensor,
    salt: torch.Tensor,
    pressure: torch.Tensor,
    *,
    eos_variant: str,
    reference_density: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(density_anomaly, density)`` from one strict CUDA launch."""
    if requires_torch_transform(temp, salt, pressure):
        from ..equation_of_state import jmd95_density_anomaly, mdjwf_density_anomaly

        kernel = {"jmd95": jmd95_density_anomaly,
                  "mdjwf": mdjwf_density_anomaly}[eos_variant]
        anomaly = kernel(temp, salt, pressure, reference_density=reference_density)
        return anomaly, anomaly + reference_density
    return eos_fwd(temp, salt, pressure, eos_variant, float(reference_density))


__all__ = [
    "_PROMOTED_DTYPES",
    "_SUPPORTED_DTYPES",
    "cuda_launch_config",
    "direct_decline_reason",
    "eos_cuda",
    "load_eos_cuda_extension",
]
