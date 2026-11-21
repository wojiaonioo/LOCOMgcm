"""Strict lazy loader for the experimental KPP bldepth CUDA extension.

The direct entry point accepts arbitrary non-negative tensor strides and never
inserts a contiguous copy or falls back. The public dispatcher in
``dispatcher.py`` owns the default-off capability gate; both decide
applicability with one predicate, :func:`direct_decline_reason`. The native
VJP differentiates seven physical inputs; geometry, tables and scalar
parameters remain fixed, and other transform modes use the Torch path.
"""

import os
import shutil
import sys
from numbers import Integral, Real
from pathlib import Path
from typing import Any, List

import torch

from locomgcm.dynamics.kpp import check_kpp_sign_zero

_HERE = Path(__file__).resolve().parent
_KEY = "kpp_bldepth_cuda"
#: The dtypes kernel.cu instantiates (AT_DISPATCH_FLOATING_TYPES).
_SUPPORTED_DTYPES = (torch.float64, torch.float32)
_DTYPE_NAMES = {torch.float64: "float64", torch.float32: "float32"}
#: The column depth and velocity-scale table shape ``extension.cpp`` accepts.
SUPPORTED_NZ = 90
SUPPORTED_TABLE_SHAPE = (892, 482)
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
        raise RuntimeError("KPP bldepth CUDA extension previously failed to load")
    from torch.utils.cpp_extension import load

    try:
        _EXTENSION = load(
            name="locomgcm_kpp_bldepth_cuda",
            sources=[str(_HERE / "extension.cpp"), str(_HERE / "kernel.cu")],
            extra_cflags=_CPP_FLAGS,
            extra_cuda_cflags=_cuda_flags(),
            with_cuda=True,
            verbose=os.environ.get("LOCOMGCM_KPP_BLDEPTH_CUDA_VERBOSE", "0") == "1",
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
            f"KPP bldepth CUDA supports float64 or float32; got {dtype}"
        )
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
    device: int,
    nz: int,
    dtype: torch.dtype,
    extension: Any | None = None,
) -> dict[str, Any]:
    """Derive once per ``(device, nz, dtype)``; later calls hit this dict.

    kernel.cu dispatches on the input dtype, so a float32 call runs
    ``kpp_bldepth_kernel<float>`` with its own register and shared-memory
    footprint.  Publishing only the extension-load float64 row would leave the
    ignition audit reading a resource basis the run never used.
    """
    name = _dtype_name(dtype)
    key = (device, nz, name)
    cached = _LAUNCH_CONFIG_CACHE.get(key)
    if cached is not None:
        return cached
    native = _load_extension() if extension is None else extension
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
    from locomgcm.accel import record_launch_decision

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
            raise RuntimeError("KPP bldepth CUDA launch configuration requires CUDA")
        device = torch.cuda.current_device()
    return _get_launch_config(device, nz, dtype)


#: One reason token per applicability rule.  The strict entry raises on these;
#: the production dispatcher records them as ``inapplicable`` and falls back.
#: Both read :func:`direct_decline_reason`, so the gate can never accept an
#: input the kernel would reject, nor decline one it would have served.
_DECLINE_MESSAGES = {
    "non_tensor": "native KPP bldepth CUDA dispatch requires tensor inputs",
    "non_cuda": "native KPP bldepth CUDA dispatch requires CUDA inputs",
    "device_mismatch": "native KPP bldepth CUDA inputs must share one device",
    "kmtj_dtype": "native KPP bldepth CUDA kmtj must be int64",
    "floating_dtype": (
        "native KPP bldepth CUDA floating inputs must be float64 or "
        "float32, all of one dtype"
    ),
    "kmtj_rank": "native KPP bldepth CUDA kmtj must be two-dimensional",
    "volume_rank": (
        "native KPP bldepth CUDA dvsq, dbloc and Ritop must be three-dimensional"
    ),
    "horizontal_rank": (
        "native KPP bldepth CUDA ustar, bo, bosol and coriol must be "
        "two-dimensional"
    ),
    "grid_rank": "native KPP bldepth CUDA zgrid and hwide must be one-dimensional",
    "table_rank": "native KPP bldepth CUDA wmt and wst must be two-dimensional",
    "unsupported_shape": (
        "native KPP bldepth CUDA supports non-empty (90, NY, NX) columns only"
    ),
    "kmtj_shape": "native KPP bldepth CUDA kmtj must have the (NY, NX) column shape",
    "volume_shape": "native KPP bldepth CUDA dbloc and Ritop must match the dvsq shape",
    "horizontal_shape": (
        "native KPP bldepth CUDA ustar, bo, bosol and coriol must have the "
        "(NY, NX) column shape"
    ),
    "grid_shape": "native KPP bldepth CUDA zgrid and hwide must have nz+2 entries",
    "table_shape": (
        "native KPP bldepth CUDA wmt and wst must have the default KPP table "
        "shape (892, 482)"
    ),
    "negative_stride": "native KPP bldepth CUDA inputs must not have negative strides",
    "limit_type": "native KPP bldepth CUDA limit_hbl_stable must be bool",
    "jwtype": "native KPP bldepth CUDA jwtype must be an integer in [1, 5]",
    "scalar_type": "native BLDEPTH requires fixed Python scalar parameters",
    "wscale_overrides": "native KPP bldepth CUDA does not support wscale overrides",
    "geometry_grad": "native BLDEPTH VJP requires fixed geometry and tables",
    "torch_transform": "native BLDEPTH supports first-order reverse only",
}


def direct_decline_reason(
    values: tuple[object, ...],
    *,
    limit_hbl_stable: object,
    jwtype: object,
    scalar_parameters: tuple[object, ...],
    wscale_overrides: dict[str, object],
) -> str | None:
    """Return the first unmet applicability rule, or ``None`` when supported.

    ``values`` is the twelve-tensor argument tuple of
    :func:`kpp_bldepth_cuda_direct`; layout and parameter rules mirror checks of
    ``extension.cpp`` ``kpp_bldepth_fwd`` or a parameter it cannot carry
    (a Tensor scalar would lose its graph in ``float()``). The native VJP serves
    physical inputs; only differentiated program geometry/tables use Torch.
    """
    if any(not isinstance(value, torch.Tensor) for value in values):
        return "non_tensor"
    reference = values[1]
    if any(value.device.type != "cuda" for value in values):
        return "non_cuda"
    if any(value.device != reference.device for value in values):
        return "device_mismatch"
    if values[0].dtype != torch.int64:
        return "kmtj_dtype"
    if reference.dtype not in _SUPPORTED_DTYPES or any(
        value.dtype != reference.dtype for value in values[1:]
    ):
        return "floating_dtype"
    if values[0].ndim != 2:
        return "kmtj_rank"
    if any(value.ndim != 3 for value in values[1:4]):
        return "volume_rank"
    if any(value.ndim != 2 for value in values[4:8]):
        return "horizontal_rank"
    if any(value.ndim != 1 for value in values[8:10]):
        return "grid_rank"
    if any(value.ndim != 2 for value in values[10:12]):
        return "table_rank"
    nz, ny, nx = values[1].shape
    if nz != SUPPORTED_NZ or ny == 0 or nx == 0:
        return "unsupported_shape"
    if values[0].shape != (ny, nx):
        return "kmtj_shape"
    if any(value.shape != (nz, ny, nx) for value in values[2:4]):
        return "volume_shape"
    if any(value.shape != (ny, nx) for value in values[4:8]):
        return "horizontal_shape"
    if any(value.shape != (nz + 2,) for value in values[8:10]):
        return "grid_shape"
    if any(value.shape != SUPPORTED_TABLE_SHAPE for value in values[10:12]):
        return "table_shape"
    # extension.cpp checks every input's strides; torch itself cannot build a
    # negative-stride tensor today, so this rule only closes the gap.
    if any(stride < 0 for value in values for stride in value.stride()):
        return "negative_stride"
    if not isinstance(limit_hbl_stable, bool):
        return "limit_type"
    if not isinstance(jwtype, Integral) or not 1 <= int(jwtype) <= 5:
        return "jwtype"
    if any(not isinstance(value, Real) for value in scalar_parameters):
        return "scalar_type"
    if wscale_overrides:
        return "wscale_overrides"
    if requires_torch_transform(*values, *scalar_parameters):
        return "torch_transform"
    # zgrid/hwide and wmt/wst are replicated program inputs, not spatial
    # state tiles. Their control scope is identical on every rank; local
    # prognostic/forcing activity must not affect native VJP admission.
    if torch.is_grad_enabled() and any(value.requires_grad for value in values[8:]):
        return "geometry_grad"
    return None


def _check_direct(
    values: tuple[object, ...],
    *,
    limit_hbl_stable: object,
    jwtype: object,
    scalar_parameters: tuple[object, ...],
    wscale_overrides: dict[str, object],
) -> None:
    reason = direct_decline_reason(
        values,
        limit_hbl_stable=limit_hbl_stable,
        jwtype=jwtype,
        scalar_parameters=scalar_parameters,
        wscale_overrides=wscale_overrides,
    )
    if reason is None:
        return
    message = _DECLINE_MESSAGES[reason]
    if reason == "wscale_overrides":
        message = f"{message}: {', '.join(sorted(wscale_overrides))}"
    raise RuntimeError(message)


def kpp_bldepth_cuda_direct(
    kmtj: torch.Tensor,
    dvsq: torch.Tensor,
    dbloc: torch.Tensor,
    Ritop: torch.Tensor,
    ustar: torch.Tensor,
    bo: torch.Tensor,
    bosol: torch.Tensor,
    coriol: torch.Tensor,
    zgrid: torch.Tensor,
    hwide: torch.Tensor,
    wmt: torch.Tensor,
    wst: torch.Tensor,
    *,
    Ricr: float,
    epsilon: float,
    Vtc: float,
    cekman: float,
    cmonob: float,
    minKPPhbl: float,
    vonk: float,
    phepsi: float = 1.0e-10,
    limit_hbl_stable: bool = True,
    jwtype: int = 2,
    penetrating_sw: bool = True,
    sign_zero: str = "signbit",
    **wscale_overrides,
):
    """Run the native kernel or raise; this strict entry point never falls back.

    ``sign_zero`` reaches the kernel as a runtime bool, so one binary serves
    both Fortran SIGN semantics (kpp.py ``KPP_SIGN_ZERO_MODES``).
    """
    sign_zero_positive = check_kpp_sign_zero(sign_zero) == "positive"
    values = (
        kmtj,
        dvsq,
        dbloc,
        Ritop,
        ustar,
        bo,
        bosol,
        coriol,
        zgrid,
        hwide,
        wmt,
        wst,
    )
    _check_direct(
        values,
        limit_hbl_stable=limit_hbl_stable,
        jwtype=jwtype,
        scalar_parameters=(
            Ricr, epsilon, Vtc, cekman, cmonob, minKPPhbl, vonk, phepsi,
        ),
        wscale_overrides=wscale_overrides,
    )
    return kpp_bldepth_fwd(
        list(values), float(Ricr), float(epsilon), float(Vtc), float(cekman),
        float(cmonob), float(minKPPhbl), float(vonk), float(phepsi),
        bool(limit_hbl_stable), int(jwtype), penetrating_sw, sign_zero_positive,
    )


@torch.library.custom_op("locomgcm::kpp_bldepth_fwd", mutates_args=())
def kpp_bldepth_fwd(
    values: List[torch.Tensor],
    Ricr: float,
    epsilon: float,
    Vtc: float,
    cekman: float,
    cmonob: float,
    minKPPhbl: float,
    vonk: float,
    phepsi: float,
    limit_hbl_stable: bool,
    jwtype: int,
    penetrating_sw: bool = True,
    sign_zero_positive: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    extension = _load_extension()
    # Publish the configuration this launch actually runs under; kernel.cu
    # dispatches on dvsq's dtype and sizes the occupancy query by its nz.
    # Cached after the first call of a given (device, nz, dtype).
    dvsq = values[1]
    device = dvsq.device.index
    _get_launch_config(
        torch.cuda.current_device() if device is None else int(device),
        int(dvsq.shape[0]),
        dvsq.dtype,
        extension,
    )
    output = tuple(
        extension.kpp_bldepth_fwd(
            *values,
            Ricr,
            epsilon,
            Vtc,
            cekman,
            cmonob,
            minKPPhbl,
            vonk,
            phepsi,
            limit_hbl_stable,
            jwtype,
            penetrating_sw,
            sign_zero_positive,
        )
    )

    _record_execution("forward")
    return output


def _record_execution(direction, outcome="hit", reason=None):
    from locomgcm.accel import current_rank, gate, record_dispatch
    if not gate(_KEY).effective_value:
        return
    record_dispatch(_KEY, outcome=outcome, variant=direction, reason=reason,
                    site=f"kpp_bldepth_{direction}", backend="cuda", rank=current_rank())


@kpp_bldepth_fwd.register_fake
def _kpp_bldepth_fwd_fake(
    values, Ricr, epsilon, Vtc, cekman, cmonob, minKPPhbl, vonk, phepsi,
    limit_hbl_stable, jwtype, penetrating_sw=True, sign_zero_positive=False,
):
    kmtj, dvsq, _, _, ustar, *_ = values
    return (
        dvsq.new_empty(ustar.shape),
        kmtj.new_empty(kmtj.shape),
        dvsq.new_empty(ustar.shape),
        dvsq.new_empty(ustar.shape),
        dvsq.new_empty(ustar.shape),
        dvsq.new_empty(dvsq.shape),
    )


@torch.library.custom_op(
    "locomgcm::kpp_bldepth_bwd", mutates_args=(),
    schema="(Tensor[] values, Tensor rib, Tensor?[] seeds, float[] parameters, "
           "bool limit, int jwtype, bool penetrating_sw, bool sign_zero_positive, "
           "bool[] needs) -> Tensor?[]",
)
def kpp_bldepth_bwd(values, rib, seeds, parameters, limit, jwtype,
                   penetrating_sw, sign_zero_positive, needs):
    try:
        output = list(_load_extension().kpp_bldepth_bwd(
            values, rib, seeds, parameters, limit, jwtype,
            penetrating_sw, sign_zero_positive, needs,
        ))
    except Exception as error:
        _record_execution("backward", "error", type(error).__name__)
        raise

    _record_execution("backward")
    return output


@kpp_bldepth_bwd.register_fake
def _kpp_bldepth_bwd_fake(values, rib, seeds, parameters, limit, jwtype,
                        penetrating_sw, sign_zero_positive, needs):
    return [values[i + 1].new_empty(values[i + 1].shape) if need else None
            for i, need in enumerate(needs)]


def _setup_context(ctx, inputs, output):
    values, *parameters, limit, jwtype, penetrating_sw, sign_zero_positive = inputs
    # Seven prognostic/forcing inputs are differentiable in this slice.
    # Shared geometry, lookup tables and scalar Tensor parameters remain Torch.
    if any(value.requires_grad for value in values[8:]):
        raise RuntimeError("native BLDEPTH VJP requires fixed geometry and tables")
    ctx.save_for_backward(*values, output[5])
    ctx.parameters, ctx.limit, ctx.jwtype = parameters, limit, jwtype
    ctx.penetrating_sw = penetrating_sw
    ctx.sign_zero_positive = sign_zero_positive
    ctx.mark_non_differentiable(output[1], output[3], output[4])
    ctx.set_materialize_grads(False)


def _backward(ctx, hbl_seed, kbl_seed, bfsfc_seed, stable_seed, casea_seed, rib_seed):
    if torch.is_grad_enabled():
        raise RuntimeError("native BLDEPTH supports first-order reverse only; "
                           "use Torch BLDEPTH for higher-order derivatives")
    *values, rib = ctx.saved_tensors
    # torch.library preserves the Tensor[] input structure in needs_input_grad.
    needs = list(ctx.needs_input_grad[0][1:8])
    gradients = kpp_bldepth_bwd(
        values, rib, [hbl_seed, bfsfc_seed, rib_seed], ctx.parameters,
        ctx.limit, ctx.jwtype, ctx.penetrating_sw, ctx.sign_zero_positive, needs,
    )
    # The dispatcher omits trailing arguments equal to their schema defaults;
    # setup_context sees them restored, but backward must match the call shape.
    return ([None, *gradients, None, None, None, None], *([None] * 12))[
        :len(ctx.needs_input_grad)
    ]


kpp_bldepth_fwd.register_autograd(_backward, setup_context=_setup_context)


def requires_torch_transform(*values):
    return torch._C._are_functorch_transforms_active() or any(
        torch.autograd.forward_ad.unpack_dual(value).tangent is not None
        for value in values if isinstance(value, torch.Tensor)
    )


__all__ = [
    "SUPPORTED_NZ",
    "SUPPORTED_TABLE_SHAPE",
    "_SUPPORTED_DTYPES",
    "cuda_launch_config",
    "direct_decline_reason",
    "kpp_bldepth_cuda_direct",
]
