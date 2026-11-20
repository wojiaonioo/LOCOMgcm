"""Strict lazy loader and torch.library entries for the fused CG2D iteration.

The kernels retain the eager pcg_solve pointwise arithmetic order. The following
compilation and division choices preserve pointwise rounding.

``-fmad=false``
    nvcc contracts ``a + b * c`` into a single FMA by default, rounding once.
    The eager reference launches a multiply kernel and then an add kernel, i.e.
    it rounds twice.  Every accumulation in ``kernel.cu`` is of that shape, so
    without the flag every kernel here differs from eager in the last place.

``-prec-div=true`` / ``-prec-sqrt=true``
    Posture, carried for consistency with every sibling loader.  No kernel in
    this package divides or takes a root; see the next paragraph.

no division in any kernel
    ``alpha = rz / pAp`` and ``beta = rz_new / rz`` (``cg2d.py:799`` /
    ``cg2d.py:832``) stay on the host as the eager 0-d tensor divisions and
    reach the kernels as device pointers.  Both are tensor/tensor, hence a true
    division; the reciprocal-multiply that eager CUDA performs for
    ``tensor / python_scalar`` never applies here, and neither does the 1-ULP
    trap that shape has sprung on two earlier kernels.

The entry points are ``torch.library.custom_op`` operators with
``register_fake`` implementations, so a compiled region sees an opaque op with a
known meta result instead of a graph break.  They are FORWARD ONLY: there is no
autograd formula, and the dispatcher's ``decline_reason`` refuses a
grad-enabled solve rather than silently detaching (see the note on
:func:`_autograd_reason` for why that check cannot live inside the ops).  Capability fallback belongs to
``dispatcher.py``; nothing in this file falls back."""

# NO ``from __future__ import annotations`` here, deliberately: it turns every
# annotation into a string, and ``torch.library.custom_op`` infers the operator
# schema from the runtime annotation objects -- with the import it refuses the
# definitions below with "Parameter phi_halo has unsupported type
# torch.Tensor".  Every sibling ``_fast.py`` omits it for the same reason.
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from defgcm.solvers._pcg_core.layout import autograd_reason, layout_reason

_HERE = Path(__file__).resolve().parent
_KEY = "cg2d_fused_pcg"
#: The scalar types ``kernel.cu`` instantiates (its ``AT_DISPATCH_FLOATING_TYPES``
#: and ``extension.cpp``'s ``CHECK_FLOATING`` name the same pair).  The
#: dispatcher imports this tuple rather than restating a dtype list, so the gate
#: and the kernel cannot disagree about what is servable.
SUPPORTED_DTYPES = (torch.float64, torch.float32)
_DTYPE_NAMES = {torch.float64: "float64", torch.float32: "float32"}
#: The kernel names ``extension.cpp::launch_config`` accepts.  The three
#: ``ssor_precond_wrap_*`` names are the in-array periodic instantiations of the
#: SSOR apply (``kernel.cu`` K3, wrap flags as template parameters);
#: ``update_xr_ring`` is the state update reading a ringed search direction
#: (K2c).
KERNEL_NAMES = (
    "spmv",
    "update_xr",
    "update_xr_ring",
    "update_xrz",
    "ssor_precond",
    "ssor_precond_wrap_x",
    "ssor_precond_wrap_y",
    "ssor_precond_wrap_xy",
    "axpy_p",
)

_EXTENSION: Any | None = None
_LOAD_FAILED = False
# -fmad=false is load-bearing, not decoration: see the module docstring.
_CUDA_FLAGS = ["-O3", "-prec-div=true", "-prec-sqrt=true", "-fmad=false"]
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
    """JIT-compile (once) and return the extension; raise the real failure.

    ``_LOAD_FAILED`` latches so a broken toolchain is not retried on every
    solve.
    """
    global _EXTENSION, _LOAD_FAILED
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_FAILED:
        raise RuntimeError("CG2D fused PCG extension previously failed to load")
    from torch.utils.cpp_extension import load

    try:
        _EXTENSION = load(
            name="defgcm_cg2d_fused",
            sources=[str(_HERE / "extension.cpp"), str(_HERE / "kernel.cu")],
            extra_cflags=_CPP_FLAGS,
            extra_cuda_cflags=_cuda_flags(),
            with_cuda=True,
            verbose=os.environ.get("DEFGCM_CG2D_FUSED_VERBOSE", "0") == "1",
        )
    except Exception:
        _LOAD_FAILED = True
        raise
    return _EXTENSION


def load_extension() -> Any:
    """Compile/load now, raising the complete build failure (registry preload)."""
    return _load_extension()


def fused_pcg_available() -> bool:
    """True when the kernels can be compiled and loaded on this machine."""
    if not torch.cuda.is_available():
        return False
    try:
        _load_extension()
    except Exception:
        return False
    return True


def cuda_launch_config(
    kernel: str, dtype: torch.dtype, device: int | None = None
) -> dict[str, Any]:
    """Return the launch decision this kernel really uses on ``device``."""
    if kernel not in KERNEL_NAMES:
        raise ValueError(f"unknown CG2D fused kernel {kernel!r}")
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported CG2D fused dtype {dtype!r}")
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CG2D fused launch configuration requires CUDA")
        device = torch.cuda.current_device()
    with torch.cuda.device(device):
        return dict(
            _load_extension().launch_config(device, kernel, _DTYPE_NAMES[dtype])
        )


# ---------------------------------------------------------------------------
# The single-source gate predicate.
#
# ``dispatcher.py`` imports these functions instead of restating the conditions,
# and each ``custom_op`` below re-runs the one that covers its own operands and
# RAISES on a reason.  One function therefore decides both "may this run" and
# "does this run": a condition can only be relaxed in one place, and relaxing it
# there relaxes the gate and the kernel entry together.
# ---------------------------------------------------------------------------


def _layout_reason(values: Sequence[Any]):
    """Device / dtype / layout reasons -- the core's, not a local copy.

    The shared predicate in _pcg_core/layout.py serves both CG2D and CG3D
    and obtains the dtype from the first operand.
    """
    return layout_reason(values, SUPPORTED_DTYPES)


_autograd_reason = autograd_reason


def _operator_shape_reason(aC: Any, aW_edge: Any, aS_edge: Any) -> str | None:
    """Shape reasons for the operator triple, shared by SPMV and the bundle."""
    if aC.ndim != 2:
        return "coefficient_rank"
    ny, nx = int(aC.shape[0]), int(aC.shape[1])
    if ny < 2 or nx < 2:
        return "degenerate_extent"
    if tuple(aW_edge.shape) != (ny, nx + 1) or tuple(aS_edge.shape) != (ny + 1, nx):
        return "coefficient_shape"
    return None


def _single_field_reason(field: Any, ny: int, nx: int) -> str | None:
    """The state layout every kernel here assumes: one ``[1, ny, nx]`` field.

    ``ndim >= 3`` is required, not merely ``numel == ny*nx``.  A bare
    ``[ny, nx]`` state is NOT equivalent to the production ``[1, ny, nx]``: the
    eager preconditioner applies branch on ``residual.ndim == pC.ndim + 1``
    (``cg2d_preconditioner.py:61``/``:74``) and the eager matvec re-adds the
    batch axis (``implicit_free_surface_cg2d.py:833``), so the two ranks take
    different eager paths and return different shapes.  Admitting the 2-D form
    here would make the fused path agree with an eager reference the production
    caller never runs.
    """
    if field.ndim < 3:
        return "state_rank"
    if tuple(field.shape[-2:]) != (ny, nx):
        return "state_shape"
    if field.numel() != ny * nx:
        return "unsupported_batch"
    return None


def spmv_decline_reason(
    phi_halo: Any, aW_edge: Any, aS_edge: Any, aC: Any
) -> str | None:
    """Why the fused SPMV cannot serve these operands (``None`` = it can)."""
    values = (phi_halo, aW_edge, aS_edge, aC)
    reason = _layout_reason(values)
    if reason is not None:
        return reason
    reason = _operator_shape_reason(aC, aW_edge, aS_edge)
    if reason is not None:
        return reason
    ny, nx = int(aC.shape[0]), int(aC.shape[1])
    if phi_halo.ndim < 3:
        return "state_rank"
    if tuple(phi_halo.shape[-2:]) != (ny + 2, nx + 2):
        return "halo_shape"
    if phi_halo.numel() != (ny + 2) * (nx + 2):
        return "unsupported_batch"
    return None


def update_decline_reason(
    x: Any, p: Any, r: Any, ap: Any, alpha: Any, precond_inv: Any | None = None
) -> str | None:
    """Why the fused x/r(/z) update cannot serve these operands."""
    values = [x, p, r, ap, alpha]
    if precond_inv is not None:
        values.append(precond_inv)
    reason = _layout_reason(values)
    if reason is not None:
        return reason
    if x.numel() == 0:
        return "degenerate_extent"
    if any(tuple(value.shape) != tuple(x.shape) for value in (p, r, ap)):
        return "state_shape"
    if alpha.numel() != 1:
        return "scalar_shape"
    if precond_inv is not None and precond_inv.numel() != x.numel():
        return "precond_shape"
    return None


def update_ring_decline_reason(
    x: Any, p_ring: Any, r: Any, ap: Any, alpha: Any
) -> str | None:
    """Why the fused x/r update from a ringed search direction cannot run.

    ``update_decline_reason``'s rules for ``x``, ``r``, ``Ap`` and ``alpha``,
    with ``x`` one ``[.., ny, nx]`` field (the kernel indexes rows) and
    ``p_ring`` that field's ``[.., ny+2, nx+2]`` ring form.
    """
    reason = _layout_reason((x, p_ring, r, ap, alpha))
    if reason is not None:
        return reason
    if x.numel() == 0:
        return "degenerate_extent"
    if any(tuple(value.shape) != tuple(x.shape) for value in (r, ap)):
        return "state_shape"
    if alpha.numel() != 1:
        return "scalar_shape"
    if x.ndim < 2:
        return "state_rank"
    ny, nx = int(x.shape[-2]), int(x.shape[-1])
    if x.numel() != ny * nx:
        return "unsupported_batch"
    if p_ring.ndim < 2 or tuple(p_ring.shape[-2:]) != (ny + 2, nx + 2):
        return "ring_shape"
    if p_ring.numel() != (ny + 2) * (nx + 2):
        return "unsupported_batch"
    return None


def ssor_decline_reason(r: Any, pC: Any, pW: Any, pS: Any) -> str | None:
    """Why the fused SSOR apply cannot serve these operands."""
    values = (r, pC, pW, pS)
    reason = _layout_reason(values)
    if reason is not None:
        return reason
    if pC.ndim != 2:
        return "coefficient_rank"
    ny, nx = int(pC.shape[0]), int(pC.shape[1])
    if ny < 2 or nx < 2:
        return "degenerate_extent"
    if tuple(pW.shape) != (ny, nx) or tuple(pS.shape) != (ny, nx):
        return "coefficient_shape"
    return _single_field_reason(r, ny, nx)


def axpy_decline_reason(z: Any, p: Any, beta: Any) -> str | None:
    """Why the fused search-direction update cannot serve these operands."""
    reason = _layout_reason((z, p, beta))
    if reason is not None:
        return reason
    if z.numel() == 0:
        return "degenerate_extent"
    if tuple(p.shape) != tuple(z.shape):
        return "state_shape"
    if beta.numel() != 1:
        return "scalar_shape"
    return None


def bundle_decline_reason(
    states: Sequence[Any],
    aW_edge: Any,
    aS_edge: Any,
    aC: Any,
    precond_operands: Sequence[Any],
    *,
    precond_layout: str = "cell",
) -> str | None:
    """Why the whole iteration bundle cannot be served for one solve.

    The per-solve predicate the dispatcher reads.  It is a strict PREFIX of the
    per-iteration predicates above -- same layout rules, same operator-shape
    rules -- so admitting a solve here cannot admit an iteration the kernels
    then refuse.

    ``states`` is EVERY externally supplied state tensor, not a sample.  The
    first revision passed ``rhs`` alone and a solve with
    ``x0.requires_grad=True`` and ``rhs.requires_grad=False`` was admitted and
    then failed at backward time.  Everything the loop
    subsequently hands the kernels -- ``x``, ``p``, ``r``, ``Ap``, ``z``, the
    halo, ``alpha``, ``beta`` -- is produced from these operands by the kernels
    themselves or by the caller's exchange, so it inherits their device, dtype
    and grad state; the per-launch predicates still re-check each one, and the
    call site now has a fallback for the case where one of them nonetheless
    refuses.

    ``precond_layout`` names the per-launch predicate the three SSOR operands
    must pass: ``"cell"`` (``pC``, ``pW``, ``pS`` all ``[ny, nx]``) for the K3
    apply, ``"edge"`` (``pC [ny, nx]``, ``pW_edge [ny, nx+1]``, ``pS_edge
    [ny+1, nx]``) for the exchanged-residual apply, which is the SPMV's own
    shape rule because that apply IS the SPMV kernel, and ``"ring"`` for the
    same apply on the tile plus a one-cell ring (``pC [ny+2, nx+2]``,
    ``pW [ny+2, nx+3]``, ``pS [ny+3, nx+2]``), again the SPMV's rule.
    """
    states = tuple(states)
    if not states:
        raise ValueError("bundle_decline_reason needs at least one state tensor")
    values = [*states, aW_edge, aS_edge, aC, *precond_operands]
    reason = _layout_reason(values)
    if reason is not None:
        return reason
    reason = _autograd_reason(values)
    if reason is not None:
        return reason
    reason = _operator_shape_reason(aC, aW_edge, aS_edge)
    if reason is not None:
        return reason
    ny, nx = int(aC.shape[0]), int(aC.shape[1])
    for state in states:
        reason = _single_field_reason(state, ny, nx)
        if reason is not None:
            return reason
    if any(tuple(state.shape) != tuple(states[0].shape) for state in states):
        return "state_shape"
    if len(precond_operands) == 3:
        if precond_layout in ("edge", "ring"):
            pC, pW_edge, pS_edge = precond_operands
            reason = _operator_shape_reason(pC, pW_edge, pS_edge)
            if reason is not None:
                return reason
            width = 0 if precond_layout == "edge" else 2
            if tuple(pC.shape) != (ny + width, nx + width):
                return "coefficient_shape"
        elif precond_layout == "cell":
            if any(tuple(value.shape) != (ny, nx) for value in precond_operands):
                return "coefficient_shape"
        else:
            raise ValueError(f"unknown SSOR operand layout {precond_layout!r}")
    elif len(precond_operands) == 1:
        if precond_operands[0].numel() != ny * nx:
            return "precond_shape"
    else:
        return "unsupported_preconditioner"
    return None


autograd_reason = _autograd_reason


def _refuse(reason: str, what: str) -> None:
    raise RuntimeError(f"native CG2D fused {what} is inapplicable: {reason}")


# ---------------------------------------------------------------------------
# torch.library entry points
# ---------------------------------------------------------------------------


@torch.library.custom_op("defgcm::cg2d_spmv", mutates_args=())
def cg2d_spmv(
    phi_halo: torch.Tensor,
    aW_edge: torch.Tensor,
    aS_edge: torch.Tensor,
    aC: torch.Tensor,
) -> torch.Tensor:
    """Five-point Helmholtz matvec from an already-exchanged centre halo."""
    reason = spmv_decline_reason(phi_halo, aW_edge, aS_edge, aC)
    if reason is not None:
        _refuse(reason, "SPMV")
    return _load_extension().cg2d_spmv_fwd(phi_halo, aW_edge, aS_edge, aC)


@cg2d_spmv.register_fake
def _cg2d_spmv_fake(phi_halo, aW_edge, aS_edge, aC):
    del aW_edge, aS_edge
    shape = (*phi_halo.shape[:-2], aC.shape[-2], aC.shape[-1])
    return phi_halo.new_empty(shape)


@torch.library.custom_op("defgcm::cg2d_update_xr", mutates_args=())
def cg2d_update_xr(
    x: torch.Tensor,
    p: torch.Tensor,
    r: torch.Tensor,
    ap: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(x + alpha*p, r - alpha*Ap)`` from one launch."""
    reason = update_decline_reason(x, p, r, ap, alpha)
    if reason is not None:
        _refuse(reason, "state update")
    x_out, r_out = _load_extension().cg2d_update_xr_fwd(x, p, r, ap, alpha)
    return x_out, r_out


@cg2d_update_xr.register_fake
def _cg2d_update_xr_fake(x, p, r, ap, alpha):
    del p, ap, alpha
    return torch.empty_like(x), torch.empty_like(r)


@torch.library.custom_op("defgcm::cg2d_update_xr_ring", mutates_args=())
def cg2d_update_xr_ring(
    x: torch.Tensor,
    p_ring: torch.Tensor,
    r: torch.Tensor,
    ap: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(x + alpha*owned(p_ring), r - alpha*Ap)`` from one launch."""
    reason = update_ring_decline_reason(x, p_ring, r, ap, alpha)
    if reason is not None:
        _refuse(reason, "ring state update")
    x_out, r_out = _load_extension().cg2d_update_xr_ring_fwd(x, p_ring, r, ap, alpha)
    return x_out, r_out


@cg2d_update_xr_ring.register_fake
def _cg2d_update_xr_ring_fake(x, p_ring, r, ap, alpha):
    del p_ring, ap, alpha
    return torch.empty_like(x), torch.empty_like(r)


@torch.library.custom_op("defgcm::cg2d_update_xrz", mutates_args=())
def cg2d_update_xrz(
    x: torch.Tensor,
    p: torch.Tensor,
    r: torch.Tensor,
    ap: torch.Tensor,
    alpha: torch.Tensor,
    precond_inv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(x + alpha*p, r - alpha*Ap, precond_inv * (r - alpha*Ap))``."""
    reason = update_decline_reason(x, p, r, ap, alpha, precond_inv)
    if reason is not None:
        _refuse(reason, "state update")
    x_out, r_out, z_out = _load_extension().cg2d_update_xrz_fwd(
        x, p, r, ap, alpha, precond_inv
    )
    return x_out, r_out, z_out


@cg2d_update_xrz.register_fake
def _cg2d_update_xrz_fake(x, p, r, ap, alpha, precond_inv):
    del p, ap, alpha, precond_inv
    return torch.empty_like(x), torch.empty_like(r), torch.empty_like(r)


@torch.library.custom_op("defgcm::cg2d_ssor_precond", mutates_args=())
def cg2d_ssor_precond(
    r: torch.Tensor, pC: torch.Tensor, pW: torch.Tensor, pS: torch.Tensor
) -> torch.Tensor:
    """Regional five-point SSOR preconditioner apply (no halo, as in eager)."""
    reason = ssor_decline_reason(r, pC, pW, pS)
    if reason is not None:
        _refuse(reason, "SSOR apply")
    return _load_extension().cg2d_ssor_precond_fwd(r, pC, pW, pS)


@cg2d_ssor_precond.register_fake
def _cg2d_ssor_precond_fake(r, pC, pW, pS):
    del pC, pW, pS
    return torch.empty_like(r)


# The in-array periodic instantiations of K3, one op each.  A flag argument on
# ``cg2d_ssor_precond`` would cost every call of the tile-local op two more
# boxed arguments (about 1.4 us on the host),
# and the choice is fixed per solve anyway.  Each is
# ``apply_cg2d_ssor_preconditioner_wrapped`` with the named axes wrapped: an
# unwrapped edge has no neighbour term, a wrapped axis reads the opposite end
# of the array.


def _ssor_precond_wrapped(r, pC, pW, pS, wrap_x: bool, wrap_y: bool):
    reason = ssor_decline_reason(r, pC, pW, pS)
    if reason is not None:
        _refuse(reason, "SSOR apply")
    return _load_extension().cg2d_ssor_precond_wrapped_fwd(
        r, pC, pW, pS, wrap_x, wrap_y
    )


@torch.library.custom_op("defgcm::cg2d_ssor_precond_wrap_x", mutates_args=())
def cg2d_ssor_precond_wrap_x(
    r: torch.Tensor, pC: torch.Tensor, pW: torch.Tensor, pS: torch.Tensor
) -> torch.Tensor:
    """K3 with the x axis wrapped inside the array."""
    return _ssor_precond_wrapped(r, pC, pW, pS, True, False)


@torch.library.custom_op("defgcm::cg2d_ssor_precond_wrap_y", mutates_args=())
def cg2d_ssor_precond_wrap_y(
    r: torch.Tensor, pC: torch.Tensor, pW: torch.Tensor, pS: torch.Tensor
) -> torch.Tensor:
    """K3 with the y axis wrapped inside the array."""
    return _ssor_precond_wrapped(r, pC, pW, pS, False, True)


@torch.library.custom_op("defgcm::cg2d_ssor_precond_wrap_xy", mutates_args=())
def cg2d_ssor_precond_wrap_xy(
    r: torch.Tensor, pC: torch.Tensor, pW: torch.Tensor, pS: torch.Tensor
) -> torch.Tensor:
    """K3 with both axes wrapped inside the array."""
    return _ssor_precond_wrapped(r, pC, pW, pS, True, True)


for _wrapped_op in (
    cg2d_ssor_precond_wrap_x,
    cg2d_ssor_precond_wrap_y,
    cg2d_ssor_precond_wrap_xy,
):

    @_wrapped_op.register_fake
    def _cg2d_ssor_precond_wrapped_fake(r, pC, pW, pS):
        del pC, pW, pS
        return torch.empty_like(r)


del _wrapped_op

#: ``(wrap_x, wrap_y)`` -> the K3 op that applies it.
SSOR_PRECOND_OPS = {
    (False, False): cg2d_ssor_precond,
    (True, False): cg2d_ssor_precond_wrap_x,
    (False, True): cg2d_ssor_precond_wrap_y,
    (True, True): cg2d_ssor_precond_wrap_xy,
}


@torch.library.custom_op("defgcm::cg2d_axpy_p", mutates_args=())
def cg2d_axpy_p(
    z: torch.Tensor, p: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Return ``z + beta * p`` from one launch."""
    reason = axpy_decline_reason(z, p, beta)
    if reason is not None:
        _refuse(reason, "search-direction update")
    return _load_extension().cg2d_axpy_p_fwd(z, p, beta)


@cg2d_axpy_p.register_fake
def _cg2d_axpy_p_fake(z, p, beta):
    del p, beta
    return torch.empty_like(z)


__all__ = [
    "KERNEL_NAMES",
    "SSOR_PRECOND_OPS",
    "autograd_reason",
    "bundle_decline_reason",
    "SUPPORTED_DTYPES",
    "axpy_decline_reason",
    "cg2d_axpy_p",
    "cg2d_spmv",
    "cg2d_ssor_precond",
    "cg2d_ssor_precond_wrap_x",
    "cg2d_ssor_precond_wrap_xy",
    "cg2d_ssor_precond_wrap_y",
    "cg2d_update_xr",
    "cg2d_update_xr_ring",
    "cg2d_update_xrz",
    "cuda_launch_config",
    "fused_pcg_available",
    "load_extension",
    "spmv_decline_reason",
    "ssor_decline_reason",
    "update_decline_reason",
    "update_ring_decline_reason",
]
