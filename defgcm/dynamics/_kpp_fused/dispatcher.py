"""Unified gate and custom-op mount for the full regional KPP CUDA scope."""

import os
import shutil
import sys
from numbers import Real
from pathlib import Path
from typing import Any, List

import torch
from defgcm.accel.tensor_context import grad_mode_requires_reference

from defgcm.accel import current_rank, dispatch_scope, gate, record_dispatch
from defgcm.dynamics.kpp import check_kpp_sign_zero


_HERE = Path(__file__).resolve().parent
_DEFGCM = _HERE.parents[1]
_KEY = "kpp_fused_cuda"
_SITE = "regional_kpp_fields"
_BACKEND = "cuda"
_SUPPORTED_DTYPES = (torch.float32, torch.float64)
_TABLE_SHAPE = (892, 482)
_SUPPORTED_NZ = 90
_EXTENSION: Any | None = None
_LOAD_FAILED = False


def _record(outcome: str, *, eos: str, reason: str | None = None) -> None:
    record_dispatch(
        _KEY,
        outcome=outcome,
        variant=eos,
        site=_SITE,
        backend=_BACKEND,
        rank=current_rank(),
        reason=reason,
    )


def direct_decline_reason(
    tensors: tuple[torch.Tensor, ...],
    *,
    eos: str,
    params: dict,
) -> str | None:
    """Single applicability predicate shared by dispatch and the strict op.

    Device-independent checks precede device, dtype and stride checks so CPU
    probes can exercise all shape and parameter rules. Gradient admission uses
    the shared program mode, independent of rank-local input activity.
    """
    if eos not in ("jmd95", "mdjwf"):
        return "eos"
    if params["use_double_diff"]:
        # the column kernel has no KPP_DOUBLEDIFF term
        return "scheme"
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        return "non_tensor"
    reference = tensors[0]
    if reference.ndim != 3 or any(value.ndim != 3 for value in tensors[1:4]):
        return "volume_rank"
    nz, ny, nx = reference.shape
    if nz != _SUPPORTED_NZ or ny < 3 or nx < 3:
        return "unsupported_shape"
    if any(value.shape != reference.shape for value in tensors[1:4]):
        return "volume_shape"
    if tensors[9].shape != reference.shape:
        return "mask_shape"
    if any(value.shape != (ny, nx) for value in tensors[4:9]):
        return "horizontal_shape"
    if tensors[10].shape != (ny, nx):
        return "kmtj_shape"
    if tensors[11].shape != (nz,):
        return "layer_shape"
    if any(value.shape != (nz + 2,) for value in tensors[12:14]):
        return "grid_shape"
    if any(value.shape != _TABLE_SHAPE for value in tensors[14:16]):
        return "table_shape"
    if tensors[16].shape != (ny, nx):
        return "qsw_shape"
    pressure = tensors[17]
    if pressure.ndim == 1:
        if pressure.shape != (nz,):
            return "pressure_shape"
    elif pressure.shape != reference.shape:
        return "pressure_shape"
    if any(value.shape != reference.shape for value in tensors[18:20]):
        return "background_shape"
    if eos == "jmd95" and params["dbloc_form"] != "divided_difference":
        # the column kernel's JMD95 dbloc is the divided difference (its MDJWF dbloc
        # the subtraction): _kpp_statekpp_fused/kernel.cu via statekpp_kernel_shim.cu
        return "dbloc_form"
    if params.get("kpp_estimate_uref", False):
        return "estimate_uref"
    if not params.get("kpp_smooth_shsq", True):
        return "smooth_shsq"
    if not params.get("kpp_smooth_dbloc", True):
        return "smooth_dbloc"
    if int(params.get("num_v_smooth_Ri", 0)) != 0:
        return "vertical_ri_smoothing"
    scalar_names = (
        "gravity", "rho_const", "Ricr", "epsilon", "Vtc", "cekman",
        "cmonob", "minKPPhbl", "vonk", "Riinfty", "BVSQcon", "difm0",
        "difs0", "dift0", "difmcon", "difscon", "diftcon", "visc_arnr",
        "cg", "conc1", "phepsi", "heat_capacity_cp",
    )
    if any(not isinstance(params.get(name), Real) for name in scalar_names):
        return "scalar_type"
    # --- device / dtype / layout screen (host-dependent; MUST stay last) ---
    # Everything above is device-independent, so validate it before checking
    # whether the tensors are on CUDA.
    if any(value.device.type != "cuda" for value in tensors):
        return "non_cuda"
    if any(value.device != reference.device for value in tensors):
        return "device_mismatch"
    floating = tensors[:10] + tensors[11:]
    if reference.dtype not in _SUPPORTED_DTYPES or any(
        value.dtype != reference.dtype for value in floating
    ):
        return "floating_dtype"
    if tensors[10].dtype != reference.dtype:
        return "kmtj_dtype"
    if any(stride < 0 for value in tensors for stride in value.stride()):
        return "negative_stride"
    if grad_mode_requires_reference():
        return "autograd"
    return None


def _find_cxx() -> str | None:
    prefix = Path(sys.prefix) / "bin"
    for name in ("x86_64-conda-linux-gnu-g++", "g++", "c++"):
        candidate = prefix / name
        if candidate.exists():
            return str(candidate)
        found = shutil.which(name)
        if found:
            return found
    return None


# nvcc flags this kernel's bitwise equality DEPENDS ON, not preferences:
#   -fmad=false     nvcc defaults to -fmad=true and would contract the
#                   multiply-then-add chains in preprocess_kernel and the
#                   wscale table interpolation into single-rounding FMAs.  The
#                   eager reference runs every multiply and add as its own ATen
#                   kernel and cannot contract, so a contracted kernel silently
#                   stops matching it.
#   -prec-div=true  keeps the tensor/tensor divides IEEE-correctly-rounded
#                   (ri_rig, ri_nsq and friends divide tensor by tensor and
#                   must round exactly as ATen's div does).
#   -prec-sqrt=true same, for the ustar sqrt chain.
# Losing any of these changes results with NO error and NO warning.  Two kernels
# (ri and finalize) additionally use explicit __d*_rn intrinsics and would
# survive the loss; preprocess and wscale would not, and hardening their ~99
# multiply terms by hand was judged riskier than pinning the flag.
REQUIRED_NVCC_FLAGS = ("-O3", "-prec-div=true", "-prec-sqrt=true", "-fmad=false")


def load_extension() -> Any:
    """Build the seven-launch extension through PyTorch's cache."""
    global _EXTENSION, _LOAD_FAILED
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_FAILED:
        raise RuntimeError("regional KPP fused extension previously failed to load")
    from torch.utils.cpp_extension import load

    cxx = _find_cxx()
    if cxx is not None:
        os.environ.setdefault("CXX", cxx)
        os.environ.setdefault("CUDAHOSTCXX", cxx)
    flags = list(REQUIRED_NVCC_FLAGS)
    if cxx is not None:
        flags.append(f"-ccbin={cxx}")
    try:
        _EXTENSION = load(
            name="defgcm_kpp_fused",
            sources=[
                str(_HERE / "extension.cpp"),
                str(_HERE / "kernel.cu"),
                # Reused package kernels enter through uniquely named shim TUs:
                # several sources all called kernel.cu collapse onto one
                # object file in the build directory.
                str(_HERE / "statekpp_kernel_shim.cu"),
                str(_HERE / "bldepth_kernel_shim.cu"),
                str(_HERE / "blmix_kernel_shim.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=flags,
            with_cuda=True,
            verbose=os.environ.get("DEFGCM_KPP_FUSED_VERBOSE", "0") == "1",
        )
    except Exception:
        _LOAD_FAILED = True
        raise
    return _EXTENSION


@torch.library.custom_op("defgcm::regional_kpp_fused", mutates_args=())
def regional_kpp_fused(
    theta: torch.Tensor,
    salt: torch.Tensor,
    u_vel: torch.Tensor,
    v_vel: torch.Tensor,
    surf_forc_t: torch.Tensor,
    surf_forc_s: torch.Tensor,
    surf_forc_u: torch.Tensor,
    surf_forc_v: torch.Tensor,
    coriol: torch.Tensor,
    mask_c: torch.Tensor,
    kmtj: torch.Tensor,
    layer_thickness: torch.Tensor,
    zgrid: torch.Tensor,
    hwide: torch.Tensor,
    wmt: torch.Tensor,
    wst: torch.Tensor,
    qsw: torch.Tensor,
    pressure: torch.Tensor,
    bg_kzs: torch.Tensor,
    bg_kzt: torch.Tensor,
    scalars: List[float],
    eos: str,
    penetrating_sw: bool = True,
    sign_zero_positive: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(scalars) != len(_SCALAR_NAMES):
        raise ValueError(f"expected {len(_SCALAR_NAMES)} KPP scalars, got {len(scalars)}")
    tensors = (
        theta, salt, u_vel, v_vel, surf_forc_t, surf_forc_s, surf_forc_u,
        surf_forc_v, coriol, mask_c, kmtj, layer_thickness, zgrid, hwide,
        wmt, wst, qsw, pressure, bg_kzs, bg_kzt,
    )
    native_params = dict(zip(_SCALAR_NAMES, scalars))
    native_params.update(
        dbloc_form="divided_difference" if eos == "jmd95" else "statekpp",
        kpp_estimate_uref=False,
        kpp_smooth_shsq=True,
        kpp_smooth_dbloc=True,
        num_v_smooth_Ri=0,
        # The op has no double-diffusion argument: the column kernel has no
        # KPP_DOUBLEDIFF term and dispatch declines use_double_diff before it
        # reaches this body.
        use_double_diff=False,
    )
    reason = direct_decline_reason(tensors, eos=eos, params=native_params)
    if reason is not None:
        raise RuntimeError(f"regional KPP fused custom op is inapplicable: {reason}")
    return tuple(load_extension().regional_kpp_fused_fwd(
        theta, salt, u_vel, v_vel, surf_forc_t, surf_forc_s, surf_forc_u,
        surf_forc_v, coriol, mask_c, kmtj, layer_thickness, zgrid, hwide,
        wmt, wst, qsw, pressure, bg_kzs, bg_kzt, scalars, eos, penetrating_sw,
        sign_zero_positive,
    ))  # type: ignore[return-value]


@regional_kpp_fused.register_fake
def _regional_kpp_fused_fake(
    theta, salt, u_vel, v_vel, surf_forc_t, surf_forc_s, surf_forc_u,
    surf_forc_v, coriol, mask_c, kmtj, layer_thickness, zgrid, hwide,
    wmt, wst, qsw, pressure, bg_kzs, bg_kzt, scalars, eos, penetrating_sw=True,
    sign_zero_positive=False,
):
    del salt, u_vel, v_vel, surf_forc_t, surf_forc_s, surf_forc_u, surf_forc_v
    del coriol, mask_c, kmtj, layer_thickness, zgrid, hwide, wmt, wst, qsw
    del pressure, bg_kzs, bg_kzt, scalars, eos
    nz, ny, nx = theta.shape
    return (
        torch.empty_like(theta), torch.empty_like(theta), torch.empty_like(theta),
        torch.empty_like(theta), theta.new_empty((ny, nx)),
    )


_SCALAR_NAMES = (
    "gravity", "rho_const", "Ricr", "epsilon", "Vtc", "cekman", "cmonob",
    "minKPPhbl", "vonk", "Riinfty", "BVSQcon", "difm0", "difs0", "dift0",
    "difmcon", "difscon", "diftcon", "visc_arnr", "cg", "conc1", "phepsi",
    "heat_capacity_cp",
)


def regional_kpp_fused_dispatch(
    tensors: tuple[torch.Tensor, ...], *, eos: str, params: dict
) -> tuple[torch.Tensor, ...] | None:
    """Return fused results, or a counted request-open fallback sentinel."""
    if not gate(_KEY).effective_value:
        return None
    with dispatch_scope(_KEY):
        reason = direct_decline_reason(tensors, eos=eos, params=params)
        if reason is not None:
            _record("inapplicable", eos=eos, reason=reason)
            return None
        try:
            output = regional_kpp_fused(
                *tensors, [float(params[name]) for name in _SCALAR_NAMES], eos,
                params["penetrating_sw"],
                check_kpp_sign_zero(params["sign_zero"]) == "positive",
            )
        except Exception as error:
            _record("error", eos=eos, reason=type(error).__name__)
            raise
        _record("hit", eos=eos)
        return output


# Debug-only stage tap.  The extension's ``regional_kpp_fused_stages`` runs the
# same body as the production forward (same seven launches, same allocation
# order, same arithmetic) but publishes every intermediate instead of only the
# five model-facing outputs, so a probe can diff the fused pipeline against the
# eager reference stage by stage and name the FIRST unequal one.  This is NOT a
# custom op: the traced op's schema and the dispatch path above are untouched,
# and nothing on the production path calls it.  The order must stay in lockstep
# with the return statement in ``extension.cpp``.
STAGE_NAMES = (
    # statekpp
    "dbloc", "dbsfc", "rho", "alpha", "beta",
    # preprocess
    "dbloc_sm", "dbloc_masked", "ritop", "dvsq", "shsq", "ustar", "bo", "bosol",
    # bldepth
    "hbl", "kbl", "bfsfc", "stable", "casea", "rib",
    # ri
    "diffus",
    # wscale
    "wm_hbl", "ws_hbl", "wm_levels", "ws_levels", "wm_d", "ws_d",
    # blmix
    "blmc", "ghat", "dkm1",
    # finalize (enhance + combine + cell-centre mask)
    "visc_out", "kzs_out", "kzt_out", "ghat_out",
    # ri_kernel sub-expression tap, one (nz+2, ny, nx) slab per slot (debug
    # entry only; the production forward passes nullptr and allocates nothing).
    "ri_tap",
)

# Slot order inside the ``ri_tap`` slab; must match the stores in ri_kernel.
RI_TAP_NAMES = (
    "ri_src", "ri_rig_a", "ri_rig_c", "ri_dz", "ri_nsq_a",
    "ri_rig", "ri_nsq", "ri_fcon", "ri_fri",
)


def regional_kpp_fused_stages(
    *native_args: Any, penetrating_sw: bool = True, sign_zero: str = "signbit"
) -> dict:
    """Run the fused scope and return ``{stage name: tensor}``.  Debug only."""
    values = load_extension().regional_kpp_fused_stages(
        *native_args, penetrating_sw, check_kpp_sign_zero(sign_zero) == "positive"
    )
    if len(values) != len(STAGE_NAMES):
        raise RuntimeError(
            f"extension returned {len(values)} stages, "
            f"STAGE_NAMES lists {len(STAGE_NAMES)}"
        )
    return dict(zip(STAGE_NAMES, values))


__all__ = [
    "REQUIRED_NVCC_FLAGS",
    "RI_TAP_NAMES",
    "STAGE_NAMES",
    "direct_decline_reason",
    "regional_kpp_fused",
    "regional_kpp_fused_dispatch",
    "regional_kpp_fused_stages",
]
