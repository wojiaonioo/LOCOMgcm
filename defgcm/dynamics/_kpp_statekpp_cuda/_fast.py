"""Strict lazy loader for the fused KPP STATEKPP CUDA extension.

The direct binding accepts float64 CUDA tensors (one dtype for all three) with
three or four dimensions and arbitrary non-negative strides; a reduced-precision
run reaches the kernel through the dispatcher, which widens to float64 and
narrows the five results back (see ``_PROMOTED_DTYPES``).  The layer pressure may be the ``(nz,)``
vector produced by :func:`defgcm.dynamics.kpp.jmd95z_layer_pressure` or any
shape that broadcasts to ``theta`` the way the eager reference broadcasts it.
Ordinary reverse mode uses a native first-order VJP. Public transforms and
Tensor gravity/reference-density parameters retain the Torch expression; a
native invocation explicitly rejects higher-order reverse differentiation.
The strict entry and the dispatcher decide applicability with one predicate,
:func:`direct_decline_reason`.

``-fmad=false`` in :data:`_CUDA_FLAGS` is a numerical requirement, not a tuning
knob: without it ptxas contracts the JMD95 polynomial into FMAs and the kernel
stops being bitwise against eager torch (see ``kernel.cu`` header).
"""

import os
import shutil
import sys
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from defgcm.accel.tensor_context import extent_mismatch

_HERE = Path(__file__).resolve().parent
_KEY = "kpp_statekpp_cuda"
#: The dtypes kernel.cu instantiates.  ``direct_decline_reason`` reads it for
#: both the strict entry and the dispatcher, so the gate cannot drift from the
#: kernel.
_SUPPORTED_DTYPES = (torch.float64,)
#: Dtypes the DISPATCHER serves by widening to float64 and narrowing back.
#: STATEKPP subtracts two density anomalies of order 10 kg/m^3 whose difference,
#: in the weakly stratified water where KPP decides the boundary-layer depth,
#: is five decades smaller; narrowing the operands first costs 1.6e-02 relative
#: on dbloc there, against 1.3e-06 in a strongly stratified column.  The eager
#: reference (``kpp.py``) therefore evaluates the whole routine in float64 and narrows only
#: its five outputs; widening here is what makes the kernel and its own fallback
#: ONE semantics.
#:
#: 🔴 The island STOPS at these five outputs.  ``_kpp_bldepth_cuda`` and
#: ``_kpp_blmix_cuda`` consume dbloc/dbsfc and are still instantiated for float32
#: (their own ``_SUPPORTED_DTYPES``), so on a float32 run BLDEPTH and BLMIX do
#: run in float32 -- deliberately, since neither subtracts near-equal operands
#: the way STATEKPP does.  "One semantics" here means the EOS/buoyancy chain,
#: not the whole KPP chain; do not read it wider than that.
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
        raise RuntimeError("KPP STATEKPP CUDA extension previously failed to load")
    from torch.utils.cpp_extension import load

    try:
        _EXTENSION = load(
            name="defgcm_kpp_statekpp_cuda",
            sources=[str(_HERE / "extension.cpp"), str(_HERE / "kernel.cu")],
            extra_cflags=_CPP_FLAGS,
            extra_cuda_cflags=_cuda_flags(),
            with_cuda=True,
            verbose=os.environ.get("DEFGCM_KPP_STATEKPP_CUDA_VERBOSE", "0") == "1",
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


def load_kpp_statekpp_cuda_extension() -> Any:
    """Compile/load the extension now, raising the complete build failure."""
    return _load_extension()


def cuda_launch_config(device: int | None = None) -> dict[str, Any]:
    """Return the native occupancy decision cached for this kernel and device."""
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("KPP STATEKPP CUDA launch configuration requires CUDA")
        device = torch.cuda.current_device()
    with torch.cuda.device(device):
        return dict(_load_extension().launch_config(device))


def normalized_inputs(
    theta: torch.Tensor, salt: torch.Tensor, layer_pressure_pa: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lift ``(theta, salt, pressure)`` to the kernel's four-dimensional view.

    The eager reference treats a **one-dimensional** pressure as the per-level
    vector (``kpp.py`` reshapes it to ``(nz, 1, 1)``, i.e. the LEVEL axis), while
    every other rank broadcasts against ``theta`` by trailing alignment.  Both
    rules are reproduced here rather than reinvented, because collapsing a
    face-major pressure field along the wrong axis returns plausible-looking
    wrong numbers instead of an error.
    """
    if theta.ndim == 3:
        theta4 = theta.unsqueeze(0)
        salt4 = salt.unsqueeze(0)
    else:
        theta4 = theta
        salt4 = salt
    nz = theta4.shape[-3]
    pressure = layer_pressure_pa
    if pressure.ndim == 1:
        pressure4 = pressure.reshape(1, nz, 1, 1)
    else:
        pressure4 = pressure.reshape(
            (1,) * (4 - pressure.ndim) + tuple(pressure.shape)
        )
    return theta4, salt4, pressure4


#: One reason token per applicability rule.  The strict entry raises on these;
#: the production dispatcher records them as ``inapplicable`` and falls back.
#: Both read :func:`direct_decline_reason`, so the gate can never accept an
#: input the kernel would reject, nor decline one it would have served.
_DECLINE_MESSAGES = {
    "non_tensor": "native KPP STATEKPP CUDA dispatch requires tensor inputs",
    "non_cuda": "native KPP STATEKPP CUDA dispatch requires CUDA inputs",
    "device_mismatch": "native KPP STATEKPP CUDA inputs must be on one device",
    "dtype": (
        "native KPP STATEKPP CUDA dispatch requires float64 inputs "
        "(reduced precision is widened by the dispatcher, never here)"
    ),
    "rank": "native KPP STATEKPP CUDA theta must be three- or four-dimensional",
    "shape": "native KPP STATEKPP CUDA theta and salt shapes must match",
    "empty_shape": "native KPP STATEKPP CUDA inputs must have positive extents",
    "scalar_type": (
        "native KPP STATEKPP CUDA gravity and rho_const must be real numbers"
    ),
    "pressure_rank": (
        "native KPP STATEKPP CUDA pressure must be one- to four-dimensional"
    ),
    "pressure_shape": "native KPP STATEKPP CUDA pressure must broadcast to theta",
    "negative_stride": (
        "native KPP STATEKPP CUDA inputs must not have negative strides"
    ),
    "torch_transform": (
        "native KPP STATEKPP CUDA has no JVP or vmap rule; transforms use the "
        "Torch expression"
    ),
}


def _real_parameters(*values: object) -> bool:
    """Whether the scalar parameters can be passed to the kernel as floats.

    A Tensor gravity or reference density may carry a graph the kernel would
    drop, so it keeps the Torch expression.
    """
    return all(isinstance(value, Real) for value in values)


def direct_decline_reason(
    theta: object,
    salt: object,
    layer_pressure_pa: object,
    *,
    gravity: object,
    rho_const: object,
    widen: bool = False,
) -> str | None:
    """Return the first unmet applicability rule, or ``None`` when supported.

    Reads the arguments the way the public entries receive them (3-D or 4-D
    theta, a ``(nz,)`` or broadcastable pressure), before
    :func:`normalized_inputs` lifts them to the kernel's 4-D view.
    ``widen=True`` is the dispatcher's question: it widens ``_PROMOTED_DTYPES``
    to float64 before the launch, so those dtypes are served as well.
    """
    values = (theta, salt, layer_pressure_pa)
    if any(not isinstance(value, torch.Tensor) for value in values):
        return "non_tensor"
    if any(value.device.type != "cuda" for value in values):
        return "non_cuda"
    if any(value.device != theta.device for value in values):
        return "device_mismatch"
    dtypes = _SUPPORTED_DTYPES + _PROMOTED_DTYPES if widen else _SUPPORTED_DTYPES
    if theta.dtype not in dtypes or any(
        value.dtype != theta.dtype for value in values
    ):
        return "dtype"
    if theta.ndim not in (3, 4):
        return "rank"
    if theta.shape != salt.shape:
        return "shape"
    if theta.numel() == 0:
        return "empty_shape"
    if not _real_parameters(gravity, rho_const):
        return "scalar_type"
    if not 1 <= layer_pressure_pa.ndim <= 4:
        return "pressure_rank"
    if layer_pressure_pa.ndim == 1:
        # The eager entry reshapes a 1-D pressure onto the LEVEL axis.
        if layer_pressure_pa.shape[0] != theta.shape[-3]:
            return "pressure_shape"
    else:
        padded = (1,) * (4 - layer_pressure_pa.ndim) + tuple(layer_pressure_pa.shape)
        target = (1,) * (4 - theta.ndim) + tuple(theta.shape)
        if any(extent_mismatch(actual, expected) for actual, expected in zip(padded, target)):
            return "pressure_shape"
    if any(stride < 0 for value in values for stride in value.stride()):
        return "negative_stride"
    if requires_torch_transform(*values):
        return "torch_transform"
    return None


def _check_direct(
    theta: object,
    salt: object,
    layer_pressure_pa: object,
    *,
    gravity: object,
    rho_const: object,
) -> None:
    reason = direct_decline_reason(
        theta, salt, layer_pressure_pa, gravity=gravity, rho_const=rho_const
    )
    if reason is not None:
        raise RuntimeError(_DECLINE_MESSAGES[reason])


@torch.library.custom_op("defgcm::kpp_statekpp_fwd", mutates_args=())
def kpp_statekpp_fwd(
    theta: torch.Tensor,
    salt: torch.Tensor,
    pressure: torch.Tensor,
    gravity: float,
    rho_const: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(_load_extension().kpp_statekpp_fwd(
        theta, salt, pressure, gravity, rho_const,
    ))


@kpp_statekpp_fwd.register_fake
def _kpp_statekpp_fwd_fake(theta, salt, pressure, gravity, rho_const):
    surface = (theta.shape[0], theta.shape[2], theta.shape[3])
    return (
        theta.new_empty(theta.shape),
        theta.new_empty(theta.shape),
        theta.new_empty(surface),
        theta.new_empty(theta.shape),
        theta.new_empty(theta.shape),
    )


@torch.library.custom_op(
    "defgcm::kpp_statekpp_bwd", mutates_args=(),
    schema="(Tensor theta, Tensor salt, Tensor pressure, Tensor? dbloc_seed, "
           "Tensor? dbsfc_seed, Tensor? rho1_seed, Tensor? alpha_seed, Tensor? beta_seed, "
           "float gravity, float rho_const, bool need_temp, bool need_salt, "
           "bool need_pressure) -> (Tensor?, Tensor?, Tensor?)",
)
def kpp_statekpp_bwd(theta, salt, pressure, dbloc_seed, dbsfc_seed, rho1_seed,
                     alpha_seed, beta_seed, gravity, rho_const,
                     need_temp, need_salt, need_pressure):
    seeds = [dbloc_seed, dbsfc_seed, rho1_seed, alpha_seed, beta_seed]
    return tuple(_load_extension().kpp_statekpp_bwd(
        theta, salt, pressure, seeds, gravity, rho_const,
        need_temp, need_salt, need_pressure,
    ))


@kpp_statekpp_bwd.register_fake
def _kpp_statekpp_bwd_fake(theta, salt, pressure, dbloc_seed, dbsfc_seed,
                          rho1_seed, alpha_seed, beta_seed, gravity, rho_const,
                          need_temp, need_salt, need_pressure):
    return tuple(theta.new_empty(theta.shape) if need else None
                 for need in (need_temp, need_salt, need_pressure))


def _setup_context(ctx, inputs, output):
    theta, salt, pressure, gravity, rho_const = inputs
    ctx.save_for_backward(theta, salt, pressure)
    ctx.gravity, ctx.rho_const = gravity, rho_const
    ctx.set_materialize_grads(False)


def _backward(ctx, dbloc_seed, dbsfc_seed, rho1_seed, alpha_seed, beta_seed):
    if torch.is_grad_enabled():
        raise RuntimeError("native STATEKPP supports first-order reverse only; "
                           "use Torch STATEKPP for higher-order derivatives")
    theta, salt, pressure = ctx.saved_tensors
    # The surface seed has no level dimension in the public forward schema.
    if rho1_seed is not None:
        rho1_seed = rho1_seed.unsqueeze(1)
    gt, gs, gp = kpp_statekpp_bwd(
        theta, salt, pressure, dbloc_seed, dbsfc_seed, rho1_seed,
        alpha_seed, beta_seed, ctx.gravity, ctx.rho_const, *ctx.needs_input_grad[:3],
    )
    return gt, gs, gp.sum_to_size(pressure.shape) if gp is not None else None, None, None


kpp_statekpp_fwd.register_autograd(_backward, setup_context=_setup_context)


def requires_torch_transform(*values):
    return torch._C._are_functorch_transforms_active() or any(
        torch.autograd.forward_ad.unpack_dual(value).tangent is not None
        for value in values if isinstance(value, torch.Tensor)
    )


def kpp_statekpp_cuda_direct(
    theta: torch.Tensor,
    salt: torch.Tensor,
    layer_pressure_pa: torch.Tensor,
    *,
    gravity: float,
    rho_const: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(dbloc, dbsfc, rho1, alpha, beta)`` from one strict CUDA launch."""
    # Routed before the device rules on purpose: Tensor parameters and
    # transforms keep the Torch expression on any device.  The two conditions
    # are the ones :func:`direct_decline_reason` declines for.
    if (not _real_parameters(gravity, rho_const)
            or requires_torch_transform(theta, salt, layer_pressure_pa)):
        from ..kpp import kpp_statekpp
        # the kernel's own expression: the divided-difference dbloc
        return kpp_statekpp(theta, salt, layer_pressure_pa, gravity=gravity,
                            rho_const=rho_const, eos_type="jmd95",
                            dbloc_form="divided_difference")
    _check_direct(theta, salt, layer_pressure_pa, gravity=gravity, rho_const=rho_const)
    theta4, salt4, pressure4 = normalized_inputs(theta, salt, layer_pressure_pa)
    dbloc, dbsfc, rho1, alpha_nr, beta_nr = kpp_statekpp_fwd(
        theta4,
        salt4,
        pressure4,
        float(gravity),
        float(rho_const),
    )
    # kpp.py: the Nrp1 virtual layer duplicates layer Nr, exactly as MITgcm does.
    alpha = torch.cat([alpha_nr, alpha_nr[..., -1:, :, :]], dim=-3)
    beta = torch.cat([beta_nr, beta_nr[..., -1:, :, :]], dim=-3)
    lead = tuple(theta.shape[:-3])
    horizontal = tuple(theta.shape[-2:])
    levels = theta.shape[-3]
    return (
        dbloc.reshape(theta.shape),
        dbsfc.reshape(theta.shape),
        rho1.reshape(lead + horizontal),
        alpha.reshape(lead + (levels + 1,) + horizontal),
        beta.reshape(lead + (levels + 1,) + horizontal),
    )


__all__ = [
    "_PROMOTED_DTYPES",
    "_SUPPORTED_DTYPES",
    "cuda_launch_config",
    "direct_decline_reason",
    "kpp_statekpp_cuda_direct",
    "load_kpp_statekpp_cuda_extension",
    "normalized_inputs",
]
