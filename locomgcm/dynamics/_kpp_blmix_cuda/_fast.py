"""Strict lazy loader for the fused KPP blmix CUDA extension.

The extension accepts arbitrary non-negative input strides and never inserts a
contiguous copy.  It never falls back internally; capability fallback belongs
to the public dispatcher. The profile has a first-order native VJP; the
public dispatcher retains the differentiable Torch wscale preparation.  The
strict entry and the dispatcher decide applicability with one predicate,
:func:`direct_decline_reason`.
"""

import os
from numbers import Real
import shutil
import sys
from pathlib import Path
from typing import Any, List

import torch

_HERE = Path(__file__).resolve().parent
_KEY = "kpp_blmix_cuda"
#: The dtypes kernel.cu instantiates (AT_DISPATCH_FLOATING_TYPES).
_SUPPORTED_DTYPES = (torch.float64, torch.float32)
#: The two argument forms :func:`direct_decline_reason` reads.  Both start with
#: the nine profile tensors (ustar, bfsfc, hbl, stable, casea, kbl, diffus,
#: zgrid, hwide).  The dispatcher's form then carries the (wmt, wst)
#: velocity-scale tables; the kernel's form carries the six velocity scales
#: the dispatcher prepares from them (wm_hbl, ws_hbl, wm_levels, ws_levels,
#: wm_d, ws_d).
_TABLE_FORM_ARITY = 11
_PREPARED_FORM_ARITY = 15
_DTYPE_NAMES = {torch.float64: "float64", torch.float32: "float32"}
#: ``(device, dtype name)`` -> the native launch configuration.  This kernel's
#: occupancy query carries no shape key, so nz is not part of the identity.
_LAUNCH_CONFIG_CACHE: dict[tuple[int, str], dict[str, Any]] = {}
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
        raise RuntimeError("KPP blmix CUDA extension previously failed to load")
    from torch.utils.cpp_extension import load

    try:
        _EXTENSION = load(
            name="locomgcm_kpp_blmix_cuda",
            sources=[str(_HERE / "extension.cpp"), str(_HERE / "kernel.cu")],
            extra_cflags=_CPP_FLAGS,
            extra_cuda_cflags=_cuda_flags(),
            with_cuda=True,
            verbose=os.environ.get("LOCOMGCM_KPP_BLMIX_CUDA_VERBOSE", "0") == "1",
        )
    except Exception:
        _LOAD_FAILED = True
        raise
    _push_launch_decision()
    return _EXTENSION


def _dtype_name(dtype: torch.dtype) -> str:
    name = _DTYPE_NAMES.get(dtype)
    if name is None:
        raise RuntimeError(f"KPP blmix CUDA supports float64 or float32; got {dtype}")
    return name


def _ledger_key(dtype: torch.dtype) -> str:
    """Keep the float64 ledger key byte-identical; suffix any other dtype."""
    return _KEY if dtype == torch.float64 else f"{_KEY}@{_dtype_name(dtype)}"


def _record_launch_decision(
    device: int, config: dict[str, Any], dtype: torch.dtype
) -> None:
    """Publish the native occupancy derivation to the unified launch view."""
    from locomgcm.accel import record_launch_decision

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
    ``kpp_blmix_kernel<float>`` with its own register and shared-memory
    footprint.  Publishing only the extension-load float64 row would leave the
    ignition audit reading a resource basis the run never used.
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
    from locomgcm.accel import record_launch_decision

    device = torch.cuda.current_device()
    record_launch_decision(
        kernel=f"{_KEY}@load",
        device=str(torch.device("cuda", device)),
        decision={"phase": "loaded"},
    )


def cuda_launch_config(
    device: int | None = None, dtype: torch.dtype = torch.float64
) -> dict[str, Any]:
    """Return the native occupancy decision cached for this kernel and device.

    ``dtype`` selects which template instantiation is measured; the two carry
    different register and shared-memory footprints, so a float32 run must read
    the float32 configuration rather than the float64 one.
    """
    _dtype_name(dtype)
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("KPP blmix CUDA launch configuration requires CUDA")
        device = torch.cuda.current_device()
    return _get_launch_config(device, dtype)


#: One reason token per applicability rule.  The strict entry raises on these;
#: the production dispatcher records them as ``inapplicable`` and falls back.
#: Both read :func:`direct_decline_reason`, so the gate can never accept an
#: input the kernel would reject, nor decline one it would have served.
_DECLINE_MESSAGES = {
    "arity": (
        "native KPP blmix CUDA dispatch requires the nine profile tensors "
        "followed by the six prepared velocity scales"
    ),
    "non_tensor": "native KPP blmix CUDA dispatch requires tensor inputs",
    "non_cuda": "native KPP blmix CUDA dispatch requires CUDA inputs",
    "device_mismatch": "native KPP blmix CUDA inputs must share one device",
    "dtype": (
        "native KPP blmix CUDA floating inputs must be float64 or float32, "
        "all of one dtype"
    ),
    "kbl_dtype": "native KPP blmix CUDA kbl must be int64",
    "scalar_type": "native BLMIX profile requires fixed scalar parameters; use Torch",
    "column_rank": "native KPP blmix CUDA column inputs must be two-dimensional",
    "profile_rank": (
        "native KPP blmix CUDA diffus must be four-dimensional, zgrid and hwide "
        "one-dimensional, and the level velocity scales three-dimensional"
    ),
    "table_rank": "native KPP blmix CUDA wmt and wst must be two-dimensional",
    "column_shape": "native KPP blmix CUDA column inputs must share the ustar shape",
    "empty_shape": "native KPP blmix CUDA column extents must be positive",
    "diffus_shape": (
        "native KPP blmix CUDA diffus must have shape (3, nz+2, NY, NX) with "
        "at least one KPP level"
    ),
    "vertical_shape": (
        "native KPP blmix CUDA zgrid and hwide must have nz+2 entries and the "
        "level velocity scales shape (nz, NY, NX)"
    ),
    "table_shape": "native KPP blmix CUDA wmt and wst must share one shape",
    "negative_stride": "native KPP blmix CUDA inputs must not have negative strides",
    "geometry_grad": "native BLMIX profile VJP requires fixed geometry",
    "torch_transform": "native BLMIX profile requires the Torch path for transforms",
}


def direct_decline_reason(
    values: tuple[object, ...],
    *,
    scalar_parameters: tuple[object, ...],
    prepared: bool,
) -> str | None:
    """Return the first unmet applicability rule, or ``None`` when supported.

    ``prepared=False`` reads the dispatcher's table form, before the velocity
    scales exist; ``prepared=True`` reads the kernel's form, the fifteen
    tensors of ``extension.cpp`` ``check_profile_inputs``.  Both forms share
    every rule on the nine profile tensors; the six prepared scales join the
    column and level rules, the two tables have rules of their own.
    ``scalar_parameters`` are the scalars the caller passes as floats.
    """
    values = tuple(values)
    if len(values) != (_PREPARED_FORM_ARITY if prepared else _TABLE_FORM_ARITY):
        return "arity"
    if not all(isinstance(value, torch.Tensor) for value in values):
        return "non_tensor"
    ustar, bfsfc, hbl, stable, casea, kbl, diffus, zgrid, hwide = values[:9]
    if prepared:
        wm_hbl, ws_hbl, wm_levels, ws_levels, wm_d, ws_d = values[9:]
        column_scales = (wm_hbl, ws_hbl, wm_d, ws_d)
        level_scales = (wm_levels, ws_levels)
        tables = ()
    else:
        column_scales = level_scales = ()
        tables = values[9:]
    if any(value.device.type != "cuda" for value in values):
        return "non_cuda"
    if any(value.device != ustar.device for value in values):
        return "device_mismatch"
    floating = values[:5] + values[6:]
    if ustar.dtype not in _SUPPORTED_DTYPES or any(
        value.dtype != ustar.dtype for value in floating
    ):
        return "dtype"
    if kbl.dtype != torch.int64:
        return "kbl_dtype"
    if not all(isinstance(value, Real) for value in scalar_parameters):
        return "scalar_type"
    columns = (ustar, bfsfc, hbl, stable, casea, kbl, *column_scales)
    if any(value.ndim != 2 for value in columns):
        return "column_rank"
    if (
        diffus.ndim != 4
        or zgrid.ndim != 1
        or hwide.ndim != 1
        or any(value.ndim != 3 for value in level_scales)
    ):
        return "profile_rank"
    if any(value.ndim != 2 for value in tables):
        return "table_rank"
    column_shape = ustar.shape
    if any(value.shape != column_shape for value in columns[1:]):
        return "column_shape"
    if len(column_shape) != 2 or any(size == 0 for size in column_shape):
        return "empty_shape"
    if diffus.shape[0] != 3 or diffus.shape[1] < 3:
        return "diffus_shape"
    if diffus.shape[-2:] != column_shape:
        return "diffus_shape"
    nz = diffus.shape[1] - 2
    if zgrid.shape != (nz + 2,) or hwide.shape != (nz + 2,):
        return "vertical_shape"
    if any(value.shape != (nz, *column_shape) for value in level_scales):
        return "vertical_shape"
    if tables and tables[0].shape != tables[1].shape:
        return "table_shape"
    if any(stride < 0 for value in values for stride in value.stride()):
        return "negative_stride"
    # These complete vertical arrays belong to the shared program, not a
    # rank-local state tile. Table derivatives remain on the Torch wscale
    # tape and reach the profile through its differentiable velocity scales.
    if torch.is_grad_enabled() and (zgrid.requires_grad or hwide.requires_grad):
        return "geometry_grad"
    if requires_torch_transform(*values):
        return "torch_transform"
    return None


def _check_direct(
    values: tuple[object, ...], *, scalar_parameters: tuple[object, ...]
) -> None:
    reason = direct_decline_reason(
        values, scalar_parameters=scalar_parameters, prepared=True
    )
    if reason is not None:
        raise RuntimeError(_DECLINE_MESSAGES[reason])


def requires_torch_transform(*values):
    return torch._C._are_functorch_transforms_active() or any(
        torch.autograd.forward_ad.unpack_dual(value).tangent is not None
        for value in values if value.is_floating_point()
    )


def kpp_blmix_cuda(
    values: tuple[torch.Tensor, ...],
    *,
    conc1: float,
    cg: float,
    phepsi: float,
) -> list[torch.Tensor]:
    """Launch the capability-checked native profile kernel exactly once."""
    _check_direct(values, scalar_parameters=(conc1, cg, phepsi))
    return kpp_blmix_fwd(list(values), float(conc1), float(cg), float(phepsi))


@torch.library.custom_op("locomgcm::kpp_blmix_fwd", mutates_args=())
def kpp_blmix_fwd(
    values: List[torch.Tensor],
    conc1: float,
    cg: float,
    phepsi: float,
) -> List[torch.Tensor]:
    extension = _load_extension()
    # Publish the configuration this launch actually runs under; kernel.cu
    # dispatches on ustar's dtype.  Cached after the first call per (device,
    # dtype).
    ustar = values[0]
    device = ustar.device.index
    _get_launch_config(
        torch.cuda.current_device() if device is None else int(device),
        ustar.dtype,
        extension,
    )
    return extension.kpp_blmix_fwd(
        *values,
        conc1,
        cg,
        phepsi,
    )


@kpp_blmix_fwd.register_fake
def _kpp_blmix_fwd_fake(values, conc1, cg, phepsi):
    ustar, diffus = values[0], values[6]
    nz = diffus.shape[1] - 2
    ny, nx = ustar.shape
    return [
        ustar.new_empty((3, nz, ny, nx)),
        ustar.new_empty((nz, ny, nx)),
        ustar.new_empty((3, ny, nx)),
    ]


@torch.library.custom_op(
    "locomgcm::kpp_blmix_bwd", mutates_args=(),
    schema="(Tensor[] values, Tensor?[] seeds, bool[] needs, float conc1, float cg, "
           "float phepsi) -> Tensor?[]",
)
def kpp_blmix_bwd(values: List[torch.Tensor], seeds: list[torch.Tensor | None],
                  needs: list[bool], conc1: float, cg: float,
                  phepsi: float) -> list[torch.Tensor | None]:
    return list(_load_extension().kpp_blmix_bwd(values, seeds, needs, conc1, cg, phepsi))


@kpp_blmix_bwd.register_fake
def _kpp_blmix_bwd_fake(values, seeds, needs, conc1, cg, phepsi):
    return [torch.empty(value.shape, device=value.device, dtype=value.dtype)
            if needed else None for value, needed in zip(values, needs)]


def _setup_profile_context(ctx, inputs, output):
    values, ctx.conc1, ctx.cg, ctx.phepsi = inputs
    ctx.save_for_backward(*values)
    ctx.set_materialize_grads(False)


def _profile_backward(ctx, seeds):
    if torch.is_grad_enabled():
        raise RuntimeError("native BLMIX profile supports first-order reverse only; use Torch for higher-order derivatives")
    values = list(ctx.saved_tensors)
    needs = list(ctx.needs_input_grad[0])
    if needs[7] or needs[8]:
        raise RuntimeError("native BLMIX profile does not differentiate geometry; use Torch")
    bars = kpp_blmix_bwd(values, seeds, needs, ctx.conc1, ctx.cg, ctx.phepsi)
    return bars, None, None, None


kpp_blmix_fwd.register_autograd(_profile_backward, setup_context=_setup_profile_context)


__all__ = [
    "_SUPPORTED_DTYPES",
    "cuda_launch_config",
    "direct_decline_reason",
    "kpp_blmix_cuda",
]
