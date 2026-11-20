"""Strict lazy loader for the fused implicit-vertical CUDA extension.

The extension accepts arbitrary non-negative input strides, including the
non-contiguous physical U/V views. It never inserts a contiguous copy and
never falls back to another implementation. The tracer backward primitive
serves the public solver's first-order physical-input autograd boundary.

Every native entry computes a Thomas error code.  Its runtime implementation
publishes the code to the numerical-failure channel
(``defgcm.deferred_checks.publish_solver_code``) under one of the families
registered below, with the decoder of the code layout the kernel defines.
Traced code never sees the channel; the returned code is the call's own.
"""

import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

from defgcm.deferred_checks import publish_solver_code, register_solver_code_family

_HERE = Path(__file__).resolve().parent
_EXTENSION: Any | None = None
_LOAD_FAILED = False
# No ``-fmad=false`` here, unlike every other CUDA loader in this repository.
# That is a recorded gap, not an oversight: the shared acceleration dispatcher
# names this file and states that changing it needs an independent verification
# round, because it moves numbers under an already published contract -- this
# core has a measured FP-order residual (max_rel 3.95e-16), not bitwise agreement,
# and the module docstring one level up says compiler contraction may reorder
# the result.  Adding the flag requires rerunning the comparison and updating
# both gradings.
_CUDA_FLAGS = ["-O3", "-prec-div=true", "-prec-sqrt=true"]
_CPP_FLAGS = ["-O3"]
_KEY = "ivadv"
_DEFAULT_NZ = 90
# Single source for the dtype gate: the kernel is instantiated for exactly these
# scalar types (kernel.cu ``implicit_vertical_{tracer,momentum}_launch``), the
# native entry re-checks the same pair, and ``implicit_vertical_fused.py``
# imports this tuple rather than restating a dtype list of its own.
SUPPORTED_DTYPES = (torch.float64, torch.float32)
_DTYPE_NAMES = {torch.float64: "float64", torch.float32: "float32"}
#: ``(device, nz, dtype name)`` -> the three native launch configurations.
_LAUNCH_CONFIG_CACHE: dict[tuple[int, int, str], dict[str, dict[str, Any]]] = {}
_BAND_LAUNCH_CONFIG_CACHE: dict[tuple[int, int, str, bool], dict[str, Any]] = {}

#: kernel.cu ``record_error``: a code is ``1 + (level * types + type - 1) *
#: columns + column`` for a failed pivot or quotient, and past every such code
#: ``1 + preflight_limit + flat result index`` for a non-finite result.
_ERROR_TYPES_PER_LEVEL = 5
_ERROR_MESSAGES = {
    1: "Thomas pivot is non-finite",
    2: "Thomas pivot is not numerically resolvable under |p_k| <= 8*eps*s_k",
    3: "Thomas elimination quotient a_k/p_k is non-finite",
    4: "Thomas elimination quotient c_prime is non-finite",
    5: "Thomas elimination quotient d_prime is non-finite",
}
#: The producer families of this extension, one per native entry.
TRACER_FAMILY = "ivadv_fused.tracer"
MOMENTUM_FAMILY = "ivadv_fused.momentum"
TRACER_BACKWARD_FAMILY = "ivadv_fused.tracer_backward"
BAND_FAMILY = "ivadv_fused.band"
BAND_TRANSPOSE_FAMILY = "ivadv_fused.band_transpose"


def _error_message_for_code(code: int, result_shape: tuple[int, ...]) -> str:
    """Decode one kernel error code against the shape of the field it solved."""
    total_columns = math.prod(result_shape[:-3] + result_shape[-2:])
    preflight_limit = result_shape[-3] * _ERROR_TYPES_PER_LEVEL * total_columns
    if not 0 < code <= preflight_limit + math.prod(result_shape):
        raise ValueError(f"code {code} is outside the layout of shape {result_shape}")
    zero_based = code - 1
    if zero_based < preflight_limit:
        stage, flat_column = divmod(zero_based, total_columns)
        level, error_offset = divmod(stage, _ERROR_TYPES_PER_LEVEL)
        error_type = error_offset + 1
        column_shape = result_shape[:-3] + result_shape[-2:]
        column = tuple(
            int(value)
            for value in torch.unravel_index(
                torch.tensor(flat_column), column_shape
            )
        )
        return f"{_ERROR_MESSAGES[error_type]} at column {column}, level k={level}"

    flat_result = zero_based - preflight_limit
    index = tuple(
        int(value)
        for value in torch.unravel_index(
            torch.tensor(flat_result), tuple(result_shape)
        )
    )
    level_position = len(index) - 3
    level = index[level_position]
    column = index[:level_position] + index[level_position + 1 :]
    return (
        "fused band/Thomas solve produced a non-finite result at "
        f"column {column}, level k={level}"
    )


for _family in (TRACER_FAMILY, MOMENTUM_FAMILY, TRACER_BACKWARD_FAMILY,
                BAND_FAMILY, BAND_TRANSPOSE_FAMILY):
    register_solver_code_family(_family, _error_message_for_code)
del _family


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
        raise RuntimeError("implicit-vertical CUDA extension previously failed to load")
    from torch.utils.cpp_extension import load

    try:
        _EXTENSION = load(
            name="defgcm_implicit_vertical_cuda",
            sources=[str(_HERE / "extension.cpp"), str(_HERE / "kernel.cu")],
            extra_cflags=_CPP_FLAGS,
            extra_cuda_cflags=_cuda_flags(),
            with_cuda=True,
            verbose=os.environ.get("DEFGCM_IVADV_CUDA_VERBOSE", "0") == "1",
        )
    except Exception:
        _LOAD_FAILED = True
        raise
    _push_launch_decisions()
    return _EXTENSION


def _dtype_name(dtype: torch.dtype) -> str:
    name = _DTYPE_NAMES.get(dtype)
    if name is None:
        raise RuntimeError(
            f"fused vertical CUDA supports float64 or float32; got {dtype}"
        )
    return name


def _ledger_key(name: str, dtype: torch.dtype) -> str:
    """Keep the float64 ledger keys byte-identical; suffix any other dtype."""
    if dtype == torch.float64:
        return f"{_KEY}/{name}"
    return f"{_KEY}/{name}@{_dtype_name(dtype)}"


def _record_launch_decisions(
    device: int, configs: Mapping[str, Mapping[str, Any]], dtype: torch.dtype
) -> None:
    """Publish each native occupancy derivation to the unified launch view."""
    from defgcm.accel import record_launch_decision

    label = str(torch.device("cuda", device))
    for name, config in configs.items():
        record_launch_decision(
            kernel=_ledger_key(name, dtype), device=label, decision=config
        )


def _get_launch_configs(
    device: int,
    nz: int,
    dtype: torch.dtype,
    extension: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Derive once per ``(device, nz, dtype)``; later solves hit this dict.

    The publication is not optional bookkeeping: kernel.cu dispatches on the
    input dtype, so a float32 solve runs ``kernel<float>``'s configuration.
    Recording only the extension-load float64 row would leave the ignition
    audit reading a resource basis the run never used.
    """
    name = _dtype_name(dtype)
    key = (device, nz, name)
    cached = _LAUNCH_CONFIG_CACHE.get(key)
    if cached is not None:
        return cached
    native = _load_extension() if extension is None else extension
    with torch.cuda.device(device):
        configs = {
            entry: dict(config)
            for entry, config in dict(native.launch_configs(device, nz, name)).items()
        }
    _LAUNCH_CONFIG_CACHE[key] = configs
    _record_launch_decisions(device, configs, dtype)
    return configs


def _push_launch_decisions() -> None:
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


def _resolve_device(rhs: torch.Tensor) -> int:
    device = rhs.device.index
    return torch.cuda.current_device() if device is None else int(device)


def cuda_launch_configs(
    device: int | None = None,
    nz: int = _DEFAULT_NZ,
    dtype: torch.dtype = torch.float64,
) -> dict[str, dict[str, Any]]:
    """Return cached occupancy decisions for all implicit-vertical kernels.

    ``dtype`` selects which template instantiation is measured; the two carry
    different register and shared-memory footprints, so a float32 run must read
    the float32 configurations rather than the float64 ones.
    """
    _dtype_name(dtype)
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "implicit-vertical CUDA launch configuration requires CUDA"
            )
        device = torch.cuda.current_device()
    return _get_launch_configs(device, nz, dtype)


def _check_direct(values: tuple[torch.Tensor, ...]) -> None:
    if any(value.device.type != "cuda" for value in values):
        raise RuntimeError("native fused vertical CUDA dispatch requires CUDA inputs")
    if values[0].dtype not in SUPPORTED_DTYPES:
        raise RuntimeError(
            "native fused vertical CUDA dispatch requires float64 or float32 inputs"
        )
    if any(value.dtype != values[0].dtype for value in values):
        raise RuntimeError(
            "native fused vertical CUDA dispatch requires one dtype across all inputs"
        )
    if any(value.ndim != 4 for value in values):
        raise RuntimeError("native fused vertical CUDA inputs must be four-dimensional")


@torch.library.custom_op("defgcm::implicit_vertical_band_solve", mutates_args=())
def implicit_vertical_band_solve(
    lower: torch.Tensor, main: torch.Tensor, upper: torch.Tensor,
    rhs: torch.Tensor, transpose: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIT-order four-band solve; native inputs retain their individual strides."""
    from ..implicit_vertical_fused import _accounted_dispatch

    variant = "band_transpose" if transpose else "band"
    with _accounted_dispatch(variant=variant, site="mitgcm_order_band_solve"):
        _check_direct((lower, main, upper, rhs))
        extension = _load_extension()
        device, nz, dtype = _resolve_device(rhs), int(rhs.shape[1]), _dtype_name(rhs.dtype)
        key = (device, nz, dtype, transpose)
        if key not in _BAND_LAUNCH_CONFIG_CACHE:
            config = dict(extension.band_launch_config(device, nz, dtype, transpose))
            _BAND_LAUNCH_CONFIG_CACHE[key] = config
            _record_launch_decisions(device, {variant: config}, rhs.dtype)
        result, error = extension.band_fwd(lower, main, upper, rhs, transpose)
        publish_solver_code(BAND_TRANSPOSE_FAMILY if transpose else BAND_FAMILY,
                            error, rhs.shape)
        return result, error


@implicit_vertical_band_solve.register_fake
def _implicit_vertical_band_solve_fake(lower, main, upper, rhs, transpose):
    # The extension allocates a contiguous output, including for strided inputs.
    return rhs.new_empty(rhs.shape), rhs.new_empty((), dtype=torch.int64)


def band_cuda(values: tuple[torch.Tensor, ...], *, transpose: bool = False):
    """Opaque first-order primitive, used inside the analytic band boundary."""
    _check_ad_inputs("band_cuda", ("lower", "main", "upper", "rhs"), values)
    return implicit_vertical_band_solve(*values, transpose)


@torch.library.custom_op("defgcm::implicit_vertical_tracer_fwd", mutates_args=())
def implicit_vertical_tracer_fwd(
    rhs: torch.Tensor,
    dt: torch.Tensor,
    diffusivity: torch.Tensor,
    recip_hfac: torch.Tensor,
    mask: torch.Tensor,
    area: torch.Tensor,
    reciprocal_area: torch.Tensor,
    recip_dr_f: torch.Tensor,
    recip_dr_c: torch.Tensor,
    w: torch.Tensor,
    advection_scheme: int,
    diagnostic_contract: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one strict tracer launch behind an opaque dispatcher boundary."""
    values = (
        rhs,
        dt,
        diffusivity,
        recip_hfac,
        mask,
        area,
        reciprocal_area,
        recip_dr_f,
        recip_dr_c,
        w,
    )
    _check_direct(values)
    extension = _load_extension()
    # Publish the configuration this launch actually runs under.  Cached after
    # the first solve of a given (device, nz, dtype), so the steady state is
    # one dict lookup.
    _get_launch_configs(_resolve_device(rhs), int(rhs.shape[1]), rhs.dtype, extension)
    result, error = extension.tracer_fwd(
        *values,
        advection_scheme,
        diagnostic_contract,
    )
    if diagnostic_contract:
        publish_solver_code(TRACER_FAMILY, error, rhs.shape)
    return result, error


@implicit_vertical_tracer_fwd.register_fake
def _implicit_vertical_tracer_fwd_fake(
    rhs,
    dt,
    diffusivity,
    recip_hfac,
    mask,
    area,
    reciprocal_area,
    recip_dr_f,
    recip_dr_c,
    w,
    advection_scheme,
    diagnostic_contract,
) -> tuple[torch.Tensor, torch.Tensor]:
    del dt, diffusivity, recip_hfac, mask, area, reciprocal_area
    del recip_dr_f, recip_dr_c, w, advection_scheme, diagnostic_contract
    # Native forward allocates a contiguous solution, even for strided inputs.
    return rhs.new_empty(rhs.shape), rhs.new_empty((), dtype=torch.int64)


@torch.library.custom_op(
    "defgcm::implicit_vertical_tracer_bwd", mutates_args=(),
    schema="(Tensor[] values, Tensor solution, Tensor grad_solution, bool[] needs, "
           "int advection_scheme, bool diagnostic_contract) -> (Tensor?[], Tensor)",
)
def implicit_vertical_tracer_bwd(values: list[torch.Tensor], solution: torch.Tensor,
                                 grad_solution: torch.Tensor, needs: list[bool],
                                 advection_scheme: int,
                                 diagnostic_contract: bool) -> tuple[list[torch.Tensor | None], torch.Tensor]:
    """Native transpose and physical pullback; caller owns broadcast reductions."""
    bars, error = _load_extension().tracer_bwd(
        values, solution, grad_solution, needs, advection_scheme, diagnostic_contract,
    )
    if diagnostic_contract:
        publish_solver_code(TRACER_BACKWARD_FAMILY, error, solution.shape)
    return list(bars), error


@implicit_vertical_tracer_bwd.register_fake
def _implicit_vertical_tracer_bwd_fake(values, solution, grad_solution, needs,
                                      advection_scheme, diagnostic_contract):
    slots = (0, 9, 1, 2, 3, 4, 5, 6, 7, 8)
    bars = [torch.empty(values[slot].shape, dtype=solution.dtype, device=solution.device)
            if needed else None for slot, needed in zip(slots, needs)]
    return bars, solution.new_empty((), dtype=torch.int64)


@torch.library.custom_op("defgcm::implicit_vertical_momentum_fwd", mutates_args=())
def implicit_vertical_momentum_fwd(
    rhs: torch.Tensor,
    dt: torch.Tensor,
    viscosity: torch.Tensor,
    recip_hfac: torch.Tensor,
    mask: torch.Tensor,
    w_left: torch.Tensor,
    w_right: torch.Tensor,
    area_left: torch.Tensor,
    area_right: torch.Tensor,
    mask_left: torch.Tensor,
    mask_right: torch.Tensor,
    reciprocal_face_area: torch.Tensor,
    recip_dr_f: torch.Tensor,
    recip_dr_c: torch.Tensor,
    upwind: bool,
    diagnostic_contract: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one strict momentum launch behind an opaque dispatcher boundary."""
    values = (
        rhs,
        dt,
        viscosity,
        recip_hfac,
        mask,
        w_left,
        w_right,
        area_left,
        area_right,
        mask_left,
        mask_right,
        reciprocal_face_area,
        recip_dr_f,
        recip_dr_c,
    )
    _check_direct(values)
    extension = _load_extension()
    _get_launch_configs(_resolve_device(rhs), int(rhs.shape[1]), rhs.dtype, extension)
    result, error = extension.momentum_fwd(
        *values,
        upwind,
        diagnostic_contract,
    )
    if diagnostic_contract:
        publish_solver_code(MOMENTUM_FAMILY, error, rhs.shape)
    return result, error


@implicit_vertical_momentum_fwd.register_fake
def _implicit_vertical_momentum_fwd_fake(
    rhs,
    dt,
    viscosity,
    recip_hfac,
    mask,
    w_left,
    w_right,
    area_left,
    area_right,
    mask_left,
    mask_right,
    reciprocal_face_area,
    recip_dr_f,
    recip_dr_c,
    upwind,
    diagnostic_contract,
) -> tuple[torch.Tensor, torch.Tensor]:
    del dt, viscosity, recip_hfac, mask, w_left, w_right
    del area_left, area_right, mask_left, mask_right, reciprocal_face_area
    del recip_dr_f, recip_dr_c, upwind, diagnostic_contract
    # Match the contiguous torch::empty allocation in momentum_fwd.
    return rhs.new_empty(rhs.shape), rhs.new_empty((), dtype=torch.int64)


def _check_ad_inputs(
    entry: str, names: tuple[str, ...], values: tuple[torch.Tensor, ...]
) -> None:
    """Admit the complete control set before entering the opaque forward op."""
    if any(value.requires_grad for value in values):
        active = ", ".join(name for name, value in zip(names, values) if value.requires_grad)
        raise RuntimeError(
            f"UNSUPPORTED_VJP: {entry} is an opaque forward; its adjoint is taped at "
            f"the tracer solver or momentum dispatch boundary and the kernel must receive detached "
            f"inputs; active inputs: {active}"
        )


def tracer_cuda(
    values: tuple[torch.Tensor, ...], *, advection_scheme: int,
    diagnostic_contract: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run direct tracer band construction and solve without intermediates."""
    _check_ad_inputs(
        "tracer_cuda",
        ("rhs", "dt", "diffusivity", "recip_hfac", "mask", "area",
         "reciprocal_area", "recip_dr_f", "recip_dr_c", "w"),
        values,
    )
    return implicit_vertical_tracer_fwd(*values, advection_scheme, diagnostic_contract)


def momentum_cuda(
    values: tuple[torch.Tensor, ...], *, upwind: bool, diagnostic_contract: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run direct momentum band construction and solve without intermediates."""
    _check_ad_inputs(
        "momentum_cuda",
        ("rhs", "dt", "viscosity", "recip_hfac", "mask", "w_left", "w_right",
         "area_left", "area_right", "mask_left", "mask_right",
         "reciprocal_face_area", "recip_dr_f", "recip_dr_c"),
        values,
    )
    return implicit_vertical_momentum_fwd(*values, upwind, diagnostic_contract)


__all__ = [
    "SUPPORTED_DTYPES",
    "band_cuda",
    "cuda_launch_configs",
    "momentum_cuda",
    "tracer_cuda",
]
