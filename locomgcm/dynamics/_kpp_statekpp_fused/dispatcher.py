"""Standalone gate and custom-op mount for the column-fused STATEKPP kernel."""

import os
import shutil
import sys
from collections import Counter
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from locomgcm.accel.tensor_context import extent_mismatch, grad_mode_requires_reference

from locomgcm.dynamics.kpp import kpp_statekpp as _eager_statekpp


_HERE = Path(__file__).resolve().parent
_ENV = "LOCOMGCM_KPP_STATEKPP_FUSED"
_SUPPORTED_DTYPES = (torch.float32, torch.float64)
_EXTENSION: Any | None = None
_LOAD_FAILED = False
_COUNTS: Counter[str] = Counter()
_TRUE = frozenset(("1", "true", "yes", "on"))
_FALSE = frozenset(("0", "false", "no", "off"))


def gate_requested() -> bool:
    """Resolve the standalone skeleton gate from its one environment key."""
    raw = os.environ.get(_ENV, "0").casefold()
    if raw in _TRUE:
        return True
    if raw in _FALSE or raw == "":
        return False
    raise ValueError(f"{_ENV} must be a boolean spelling, got {raw!r}")


def direct_decline_reason(
    theta: object,
    salt: object,
    layer_pressure_pa: object,
    gravity: object,
    rho_const: object,
) -> str | None:
    """Single capability predicate used by both the gate and direct op."""
    values = (theta, salt, layer_pressure_pa)
    if any(not isinstance(value, torch.Tensor) for value in values):
        return "non_tensor"
    assert isinstance(theta, torch.Tensor)
    assert isinstance(salt, torch.Tensor)
    assert isinstance(layer_pressure_pa, torch.Tensor)
    if any(value.device.type != "cuda" for value in values):
        return "non_cuda"
    if any(value.device != theta.device for value in values):
        return "device_mismatch"
    if theta.dtype not in _SUPPORTED_DTYPES or any(
        value.dtype != theta.dtype for value in values
    ):
        return "dtype"
    if theta.ndim not in (3, 4):
        return "rank"
    if theta.shape != salt.shape:
        return "shape"
    if theta.numel() == 0:
        return "empty_shape"
    if not all(isinstance(value, Real) for value in (gravity, rho_const)):
        return "scalar_type"
    if not 1 <= layer_pressure_pa.ndim <= 4:
        return "pressure_rank"
    if layer_pressure_pa.ndim == 1:
        if layer_pressure_pa.shape[0] != theta.shape[-3]:
            return "pressure_shape"
    else:
        pressure_shape = (1,) * (4 - layer_pressure_pa.ndim) + tuple(
            layer_pressure_pa.shape
        )
        theta_shape = (1,) * (4 - theta.ndim) + tuple(theta.shape)
        if any(
            extent_mismatch(actual, expected)
            for actual, expected in zip(pressure_shape, theta_shape)
        ):
            return "pressure_shape"
    if any(stride < 0 for value in values for stride in value.stride()):
        return "negative_stride"
    if grad_mode_requires_reference():
        return "grad"
    return None


def _normalized_inputs(
    theta: torch.Tensor,
    salt: torch.Tensor,
    pressure: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    theta4 = theta.unsqueeze(0) if theta.ndim == 3 else theta
    salt4 = salt.unsqueeze(0) if salt.ndim == 3 else salt
    if pressure.ndim == 1:
        pressure4 = pressure.reshape(1, theta.shape[-3], 1, 1)
    else:
        pressure4 = pressure.reshape((1,) * (4 - pressure.ndim) + pressure.shape)
    return theta4, salt4, pressure4


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


def load_extension() -> Any:
    """Build/load the CUDA extension through PyTorch's default cache root."""
    global _EXTENSION, _LOAD_FAILED
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_FAILED:
        raise RuntimeError("STATEKPP fused extension previously failed to load")
    from torch.utils.cpp_extension import load

    cxx = _find_conda_cxx()
    if cxx is not None:
        os.environ.setdefault("CXX", cxx)
        os.environ.setdefault("CUDAHOSTCXX", cxx)
    cuda_flags = [
        "-O3",
        "-prec-div=true",
        "-prec-sqrt=true",
        "-fmad=false",
    ]
    if cxx is not None:
        cuda_flags.append(f"-ccbin={cxx}")
    try:
        _EXTENSION = load(
            name="locomgcm_kpp_statekpp_fused",
            sources=[str(_HERE / "extension.cpp"), str(_HERE / "kernel.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=cuda_flags,
            with_cuda=True,
            verbose=os.environ.get("LOCOMGCM_KPP_STATEKPP_FUSED_VERBOSE", "0") == "1",
        )
    except Exception:
        _LOAD_FAILED = True
        raise
    return _EXTENSION


@torch.library.custom_op("locomgcm::statekpp_fused", mutates_args=())
def statekpp_fused(
    theta: torch.Tensor,
    salt: torch.Tensor,
    layer_pressure_pa: torch.Tensor,
    gravity: float,
    rho_const: float,
    eos: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Execute one strict CUDA launch; no fallback and no autograd."""
    if eos not in ("jmd95", "mdjwf"):
        raise ValueError(f"eos must be exactly 'jmd95' or 'mdjwf', got {eos!r}")
    reason = direct_decline_reason(
        theta, salt, layer_pressure_pa, gravity, rho_const
    )
    if reason is not None:
        raise RuntimeError(f"STATEKPP fused custom op is inapplicable: {reason}")
    theta4, salt4, pressure4 = _normalized_inputs(
        theta, salt, layer_pressure_pa
    )
    result = load_extension().statekpp_fused_fwd(
        theta4, salt4, pressure4, float(gravity), float(rho_const), eos
    )
    if theta.ndim == 3:
        return tuple(value.squeeze(0) for value in result)  # type: ignore[return-value]
    return tuple(result)  # type: ignore[return-value]


@statekpp_fused.register_fake
def _statekpp_fused_fake(
    theta: torch.Tensor,
    salt: torch.Tensor,
    layer_pressure_pa: torch.Tensor,
    gravity: float,
    rho_const: float,
    eos: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    del salt, layer_pressure_pa, gravity, rho_const, eos
    lead = theta.shape[:-3]
    levels, ny, nx = theta.shape[-3:]
    return (
        torch.empty_like(theta),
        torch.empty_like(theta),
        theta.new_empty(lead + (ny, nx)),
        theta.new_empty(lead + (levels + 1, ny, nx)),
        theta.new_empty(lead + (levels + 1, ny, nx)),
    )


def statekpp_fused_dispatch(
    theta: torch.Tensor,
    salt: torch.Tensor,
    layer_pressure_pa: torch.Tensor,
    *,
    gravity: float,
    rho_const: float,
    eos_type: str,
    dbloc_form: str,
) -> tuple[torch.Tensor, ...]:
    """Counting fallback shell for the standalone, non-integrated ring.

    The kernel's JMD95 dbloc is the divided difference and its MDJWF dbloc the
    subtraction, so a JMD95 call with ``dbloc_form="statekpp"`` falls back."""
    if eos_type not in ("jmd95", "mdjwf"):
        raise ValueError(
            f"eos_type must be exactly 'jmd95' or 'mdjwf', got {eos_type!r}"
        )
    if not gate_requested():
        _COUNTS["disabled"] += 1
        return _eager_statekpp(
            theta,
            salt,
            layer_pressure_pa,
            gravity=gravity,
            rho_const=rho_const,
            eos_type=eos_type,
            dbloc_form=dbloc_form,
        )
    reason = direct_decline_reason(
        theta, salt, layer_pressure_pa, gravity, rho_const
    )
    if reason is None and eos_type == "jmd95" and dbloc_form != "divided_difference":
        reason = "dbloc_form"
    if reason is not None:
        _COUNTS[f"fallback:{reason}"] += 1
        return _eager_statekpp(
            theta,
            salt,
            layer_pressure_pa,
            gravity=gravity,
            rho_const=rho_const,
            eos_type=eos_type,
            dbloc_form=dbloc_form,
        )
    try:
        result = statekpp_fused(
            theta,
            salt,
            layer_pressure_pa,
            float(gravity),
            float(rho_const),
            eos_type,
        )
    except Exception:
        _COUNTS["error"] += 1
        raise
    _COUNTS["requested-hit"] += 1
    return result


def dispatch_counts() -> dict[str, int]:
    return dict(_COUNTS)


def reset_dispatch_counts() -> None:
    _COUNTS.clear()


__all__ = [
    "direct_decline_reason",
    "dispatch_counts",
    "gate_requested",
    "load_extension",
    "reset_dispatch_counts",
    "statekpp_fused",
    "statekpp_fused_dispatch",
]
