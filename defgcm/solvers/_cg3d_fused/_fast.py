"""Strict lazy loader and torch.library entries for the fused CG3D iteration.

The standard operators retain the eager pointwise arithmetic order. Three
additional operators reduce inner products inside update kernels; graph.py
uses these operators, whose summation order differs from the eager reduction.
The following compilation and division choices preserve pointwise rounding.

``-fmad=false``
    nvcc contracts ``a + b * c`` into a single FMA by default, rounding once.
    The eager reference launches a multiply kernel and then an add kernel, i.e.
    it rounds twice.  Every accumulation in ``kernel.cu`` is of that shape, so
    without the flag every kernel here differs from eager in the last place.

``-prec-div=true`` / ``-prec-sqrt=true``
    Posture, carried for consistency with every sibling loader.  No kernel in
    this package divides or takes a root.

no division in any kernel
    ``cg_beta = eta_qr / eta_qr_prev`` and ``alpha = eta_qr / alpha_denom``
    (``cg3d.py:242`` / ``:261``) stay on the host as the eager 0-d tensor
    divisions and reach the kernels as device pointers.  Both are
    tensor/tensor, hence a true division; the reciprocal-multiply that eager
    CUDA performs for ``tensor / python_scalar`` never applies here, and neither
    does the 1-ULP trap that shape has sprung on two earlier kernels.

**No topology**.  The predicates below name no grid
family and no rank count; they ask only what the kernels can index.  The leading
extent ``nf`` is a plain outer dimension -- 1 for a regional field, ``n_faces``
for a face-leading one -- and coefficients may either broadcast over it or be
resolved per index."""

# NO ``from __future__ import annotations`` here, deliberately: it turns every
# annotation into a string, and ``torch.library.custom_op`` infers the operator
# schema from the runtime annotation objects.  Every sibling ``_fast.py`` omits
# it for the same reason.
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from defgcm.solvers._pcg_core.layout import autograd_reason, layout_reason

_HERE = Path(__file__).resolve().parent
_KEY = "cg3d_fused_pcg"
#: The scalar types ``kernel.cu`` instantiates.  The dispatcher imports this
#: tuple rather than restating a dtype list, so the gate and the kernel cannot
#: disagree about what is servable.
SUPPORTED_DTYPES = (torch.float64, torch.float32)
_DTYPE_NAMES = {torch.float64: "float64", torch.float32: "float32"}
#: The kernel names ``extension.cpp::launch_config`` accepts.
KERNEL_NAMES = ("spmv_halo", "axpy_s", "axpy_s_pad", "update_xr",
                "thomas_mask")
#: The dot-folding instantiations of three of those sources.  Each is a
#: DIFFERENT kernel image -- its own register count, its own static shared
#: memory for the block reduction, hence its own occupancy answer -- so each is
#: reported under its own name.  Kept separate from ``KERNEL_NAMES`` because
#: these, unlike every kernel in that tuple, do allocate static shared memory.
DOT_KERNEL_NAMES = ("spmv_halo_dot", "update_xr_dot", "thomas_mask_dot")

_EXTENSION: Any = None
_LOAD_FAILED = False
# -fmad=false is load-bearing, not decoration: see the module docstring.
_CUDA_FLAGS = ["-O3", "-prec-div=true", "-prec-sqrt=true", "-fmad=false"]
_CPP_FLAGS = ["-O3"]


def _find_conda_cxx():
    prefix = Path(sys.prefix) / "bin"
    for name in ("x86_64-conda-linux-gnu-g++", "g++", "c++"):
        candidate = prefix / name
        if candidate.exists():
            return str(candidate)
        found = shutil.which(name)
        if found:
            return found
    return None


def _cuda_flags():
    cxx = _find_conda_cxx()
    if cxx is None:
        return list(_CUDA_FLAGS)
    os.environ.setdefault("CXX", cxx)
    os.environ.setdefault("CUDAHOSTCXX", cxx)
    return [*_CUDA_FLAGS, f"-ccbin={cxx}"]


def _load_extension() -> Any:
    """JIT-compile (once) and return the extension; raise the real failure.

    ``_LOAD_FAILED`` latches so a broken toolchain is not retried on every solve.
    """
    global _EXTENSION, _LOAD_FAILED
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_FAILED:
        raise RuntimeError("CG3D fused PCG extension previously failed to load")
    from torch.utils.cpp_extension import load

    try:
        _EXTENSION = load(
            name="defgcm_cg3d_fused",
            sources=[str(_HERE / "extension.cpp"), str(_HERE / "kernel.cu")],
            extra_cflags=_CPP_FLAGS,
            extra_cuda_cflags=_cuda_flags(),
            with_cuda=True,
            verbose=os.environ.get("DEFGCM_CG3D_FUSED_VERBOSE", "0") == "1",
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


def cuda_launch_config(kernel: str, dtype: torch.dtype, device=None) -> dict:
    """Return the launch decision this kernel really uses on ``device``."""
    if kernel not in KERNEL_NAMES + DOT_KERNEL_NAMES:
        raise ValueError(f"unknown CG3D fused kernel {kernel!r}")
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported CG3D fused dtype {dtype!r}")
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CG3D fused launch configuration requires CUDA")
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
# "does this run".
# ---------------------------------------------------------------------------


def _coefficient_face_reason(coefficient, nf, per_face, ):
    """``None`` when the coefficient broadcasts over -- or resolves -- ``nf``."""
    shape = tuple(int(extent) for extent in coefficient.shape)
    if shape == tuple(per_face):
        return None
    if len(shape) == len(per_face) + 1 and shape[0] == nf and shape[1:] == tuple(per_face):
        return None
    return "coefficient_shape"


#: Ledger reasons for the two mask conditions of this package.
#: Both are spellings ``defgcm/accel/runtime.py``'s ``Reason`` enum already
#: accepts -- that vocabulary is closed, and a reason it does not know raises
#: at ``record_dispatch`` rather than being recorded, so a decline for an
#: unregistered spelling would crash the solve it was trying to describe.
#: ``dtype`` and ``mask_shape`` say what is wrong at the granularity the enum
#: offers; the precise condition is in the raised message, which the ledger
#: truncates into the same slot.
_MASK_DTYPE_REASON = "dtype"
_MASK_VALUES_REASON = "mask_shape"


def _assert_reasons_are_recordable() -> None:
    """Fail at IMPORT if a decline reason here is not one the ledger accepts.

    ``record_dispatch`` raises on an unregistered reason, so a decline spelled
    with a new word does not merely record badly -- it crashes the solve it was
    describing, at the moment the accelerator was trying to step aside
    gracefully.  That is the worst possible time to discover a typo, and it
    cost three GPU jobs before this check existed.  Checking at import turns it
    into an immediate, local failure instead.
    """
    from defgcm.accel.runtime import _REASONS

    declared = {_MASK_DTYPE_REASON, _MASK_VALUES_REASON}
    unknown = declared - _REASONS
    if unknown:
        raise RuntimeError(
            f"CG3D fused decline reasons {sorted(unknown)} are not in the "
            "accelerator ledger's vocabulary; record_dispatch would raise on "
            "them. Add them to defgcm/accel/runtime.py::Reason or reuse an "
            "existing spelling."
        )


_assert_reasons_are_recordable()


def mask_binary_reason(mask) -> str | None:
    """A reason unless every mask value is exactly 0 or 1.

    The uint8 mask is equivalent to a float multiplicative mask only when
    its values are 0 and 1. The stepper builds it as a
    product of two indicator fields (``implicit_free_surface.py:3869-3879``),
    and ``float -> uint8`` truncates toward zero, so a 0.5 would silently
    become a 0 rather than half-weighting the cell.  A mask that is not binary
    is refused at admission, with a reason in the ledger, rather than quietly
    given a different meaning.

    Asked once per SOLVE (it reads the whole field), not once per launch.
    """
    if not isinstance(mask, torch.Tensor):
        return "non_tensor"
    if mask.dtype == torch.uint8:
        # Already converted; a byte mask can only hold what was put in it, and
        # what put it there is the conversion below, which checks.
        return None
    # Both conditions in ONE pass over the field.  ``-0.0 != 0`` is FALSE, so
    # the value test alone admits it -- and ``x * -0.0`` and ``x * +0.0``
    # differ in the sign bit, which uint8 cannot carry and which
    # ``torch.equal`` cannot see, so a mask holding -0.0 would flip the sign
    # of zeros on dry points with every comparison in this suite staying green
    #.  Production never produces one -- the mask is a
    # product of two indicator fields -- so this only ever refuses.
    #
    # Fused rather than a second ``.any()``: this runs once per solve on the
    # full field, and a separate ``signbit`` pass measured 0.066 F per
    # iteration of added DRAM traffic on the 800x800x90 case, which is real
    # bandwidth spent to check something that is almost always fine.
    bad = (mask != 0) & (mask != 1)
    bad |= torch.signbit(mask)
    if bool(bad.any()):
        return _MASK_VALUES_REASON
    return None


def to_kernel_mask(mask: torch.Tensor) -> torch.Tensor:
    """The uint8 mask the kernels take, from the caller's float mask.

    One conversion point, called once per solve by the dispatcher.  Refuses
    rather than truncating -- see :func:`mask_binary_reason`.
    """
    if mask.dtype == torch.uint8:
        return mask
    reason = mask_binary_reason(mask)
    if reason is not None:
        _refuse(reason, "mask conversion")
    return mask.to(torch.uint8).contiguous()


def _mask_kind_reason(reference, mask) -> str | None:
    """The dtype/device/layout half of :func:`mask_reason`, without the shape.

    Split out for the SPMV, where the mask defines the interior extents rather
    than matching a state tensor -- its shape is checked against the padded
    block instead.  Everything else about a mask is asked here, once.
    """
    if not isinstance(mask, torch.Tensor):
        return "non_tensor"
    if mask.device.type != "cuda":
        return "non_cuda"
    if mask.dtype != torch.uint8:
        return _MASK_DTYPE_REASON
    if not mask.is_contiguous():
        return "non_contiguous"
    if mask.device != reference.device:
        return "device_mismatch"
    return None


def mask_reason(state, mask) -> str | None:
    """Why ``mask`` cannot serve as this state's mask.

    The mask is UINT8 here, not the state dtype: every kernel
    multiplies a fully formed value by 0 or 1, and a byte carries that exactly
    while moving an eighth of the bytes a float64 mask moved.  It is therefore
    excluded from ``layout_reason``'s uniform-dtype check and asked about
    separately -- ONE function, so the five entry points cannot disagree about
    what a mask is.

    The dispatcher converts once per solve, asserting the values really are
    binary first (``mask_not_binary``); a mask that arrives here still floating
    means that conversion was skipped, which is a defect, not a fallback.
    """
    reason = _mask_kind_reason(state, mask)
    if reason is not None:
        return reason
    if tuple(mask.shape) != tuple(state.shape):
        return "mask_shape"
    return None


def state_reason(state, mask) -> str | None:
    """The state layout every kernel here assumes: ``[nf, nz, ny, nx]``."""
    if state.ndim != 4:
        return "state_rank"
    if tuple(state.shape) != tuple(mask.shape):
        return "mask_shape"
    if state.numel() == 0:
        return "degenerate_extent"
    return None


def operator_shape_reason(aW, aS, aV, aC, mask) -> str | None:
    """Shape reasons for the seven-point coefficient set."""
    if mask.ndim != 4:
        return "mask_shape"
    nf, nz, ny, nx = (int(extent) for extent in mask.shape)
    if nf < 1 or nz < 1 or ny < 1 or nx < 1:
        return "degenerate_extent"
    # Unlike the 2-D SSOR kernel nothing here requires nx >= 2 or ny >= 2: the
    # padded block makes a one-cell extent well defined once the four
    # horizontal guards are gone.
    for coefficient, per_face, in (
        (aC, (nz, ny, nx)),
        (aV, (nz, ny, nx)),
        (aW, (nz, ny, nx + 1)),
        (aS, (nz, ny + 1, nx)),
    ):
        reason = _coefficient_face_reason(coefficient, nf, per_face)
        if reason is not None:
            return reason
    if (aC.ndim == 4) != (aV.ndim == 4):
        return "coefficient_shape"
    return None


def spmv_decline_reason(phi_halo, aW, aS, aV, aC, mask) -> str | None:
    """Why the fused seven-point apply cannot serve these operands."""
    reason = layout_reason((phi_halo, aW, aS, aV, aC), SUPPORTED_DTYPES)
    if reason is not None:
        return reason
    reason = operator_shape_reason(aW, aS, aV, aC, mask)
    if reason is not None:
        return reason
    nf, nz, ny, nx = (int(extent) for extent in mask.shape)
    if phi_halo.ndim != 4:
        return "state_rank"
    if tuple(phi_halo.shape) != (nf, nz, ny + 2, nx + 2):
        return "halo_shape"
    # Here the mask defines the interior extents rather than matching a state
    # tensor, so the shape half of ``mask_reason`` is the check just made
    # above; what remains is its dtype/device/layout half.
    return _mask_kind_reason(phi_halo, mask)


def axpy_decline_reason(q, s, beta, mask) -> str | None:
    """Why the fused search-direction update cannot serve these operands."""
    reason = layout_reason((q, s, beta), SUPPORTED_DTYPES)
    if reason is not None:
        return reason
    reason = mask_reason(q, mask)
    if reason is not None:
        return reason
    if q.numel() == 0:
        return "degenerate_extent"
    if tuple(s.shape) != tuple(q.shape) or tuple(mask.shape) != tuple(q.shape):
        return "state_shape"
    if beta.numel() != 1:
        return "scalar_shape"
    return None


def axpy_pad_decline_reason(q, beta, mask, padded, halo) -> str | None:
    """Why the in-place padded search-direction update cannot serve these.

    The same conditions ``axpy_decline_reason`` asks, with ``s`` replaced by
    the padded buffer that holds it -- one predicate per entry point, and the
    shared parts (layout, mask, scalar) are the shared functions.
    """
    reason = layout_reason((q, beta, padded), SUPPORTED_DTYPES)
    if reason is not None:
        return reason
    reason = mask_reason(q, mask)
    if reason is not None:
        return reason
    if q.numel() == 0:
        return "degenerate_extent"
    if q.ndim != 4 or padded.ndim != 4:
        return "state_rank"
    nf, nz, ny, nx = (int(extent) for extent in q.shape)
    if tuple(padded.shape) != (nf, nz, ny + 2 * halo, nx + 2 * halo):
        return "halo_shape"
    if beta.numel() != 1:
        return "scalar_shape"
    return None


def update_decline_reason(x, s, r, qmat, alpha, mask, *, s_halo=0) -> str | None:
    """Why the fused x/r update cannot serve these operands.

    ``s_halo`` > 0 says ``s`` is a padded buffer rather than a dense field of
    the state's shape; the kernel offsets its index by that width.  Everything
    else is unchanged, so one predicate still covers both callers.
    """
    reason = layout_reason((x, s, r, qmat, alpha), SUPPORTED_DTYPES)
    if reason is not None:
        return reason
    reason = mask_reason(x, mask)
    if reason is not None:
        return reason
    if x.numel() == 0:
        return "degenerate_extent"
    if any(tuple(value.shape) != tuple(x.shape) for value in (r, qmat, mask)):
        return "state_shape"
    if s_halo:
        if x.ndim != 4 or s.ndim != 4:
            return "state_rank"
        nf, nz, ny, nx = (int(extent) for extent in x.shape)
        if tuple(s.shape) != (nf, nz, ny + 2 * s_halo, nx + 2 * s_halo):
            return "halo_shape"
    elif tuple(s.shape) != tuple(x.shape):
        return "state_shape"
    if alpha.numel() != 1:
        return "scalar_shape"
    return None


def thomas_decline_reason(zMC, zML, zMU, r, mask) -> str | None:
    """Why the fused Thomas apply cannot serve these operands."""
    reason = layout_reason((zMC, zML, zMU, r), SUPPORTED_DTYPES)
    if reason is not None:
        return reason
    reason = mask_reason(r, mask)
    if reason is not None:
        return reason
    reason = state_reason(r, mask)
    if reason is not None:
        return reason
    nf, nz, ny, nx = (int(extent) for extent in r.shape)
    for factor in (zMC, zML, zMU):
        sub = _coefficient_face_reason(factor, nf, (nz, ny, nx))
        if sub is not None:
            return sub
    if len({factor.ndim for factor in (zMC, zML, zMU)}) != 1:
        return "coefficient_shape"
    return None


def bundle_decline_reason(
    states: Sequence[Any],
    aW,
    aS,
    aV,
    aC,
    mask,
    thomas: Sequence[Any],
) -> str | None:
    """Why the whole iteration bundle cannot be served for one solve.

    The per-solve predicate the dispatcher reads.  It is a strict PREFIX of the
    per-iteration predicates above -- same layout rules, same shape rules -- so
    admitting a solve here cannot admit an iteration the kernels then refuse.

    ``states`` is EVERY externally supplied state tensor, not a sample: the cg2d
    sibling was admitted on ``rhs`` alone and a solve with
    ``x0.requires_grad=True`` reached kernels that have no backward formula.
    """
    states = tuple(states)
    if not states:
        raise ValueError("bundle_decline_reason needs at least one state tensor")
    # This predicate runs BEFORE the dispatcher converts the mask, so here the
    # mask is still the caller's float tensor and belongs in the uniform-dtype
    # check with everything else.  ``mask_reason`` is about the converted mask
    # the kernels receive, and is asked per launch; asking it here would refuse
    # every solve.
    values = [*states, aW, aS, aV, aC, mask, *thomas]
    reason = layout_reason(values, SUPPORTED_DTYPES)
    if reason is not None:
        return reason
    reason = autograd_reason(values)
    if reason is not None:
        return reason
    reason = mask_binary_reason(mask)
    if reason is not None:
        return reason
    for state in states:
        reason = state_reason(state, mask)
        if reason is not None:
            return reason
    reason = operator_shape_reason(aW, aS, aV, aC, mask)
    if reason is not None:
        return reason
    # The SHAPE half of the Thomas predicate.  Its mask half asks for the
    # converted uint8 mask, which does not exist yet at admission time -- so
    # asking the whole predicate here would decline every solve for ``dtype``.
    # The mask itself was already checked above, twice: ``layout_reason`` for
    # its dtype/device/layout and ``mask_binary_reason`` for its values.
    reason = state_reason(states[0], mask)
    if reason is not None:
        return reason
    nf, nz, ny, nx = (int(extent) for extent in states[0].shape)
    for factor in thomas:
        sub = _coefficient_face_reason(factor, nf, (nz, ny, nx))
        if sub is not None:
            return sub
    if len({factor.ndim for factor in thomas}) != 1:
        return "coefficient_shape"
    return None


def _refuse(reason: str, what: str) -> None:
    raise RuntimeError(f"native CG3D fused {what} is inapplicable: {reason}")


# ---------------------------------------------------------------------------
# torch.library entry points
# ---------------------------------------------------------------------------


@torch.library.custom_op("defgcm::cg3d_spmv_halo", mutates_args=())
def cg3d_spmv_halo(
    phi_halo: torch.Tensor,
    aW: torch.Tensor,
    aS: torch.Tensor,
    aV: torch.Tensor,
    aC: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Seven-point apply from a padded block, mask on the store."""
    reason = spmv_decline_reason(phi_halo, aW, aS, aV, aC, mask)
    if reason is not None:
        _refuse(reason, "SPMV")
    return _load_extension().cg3d_spmv_halo_fwd(phi_halo, aW, aS, aV, aC, mask)


@cg3d_spmv_halo.register_fake
def _cg3d_spmv_halo_fake(phi_halo, aW, aS, aV, aC, mask):
    del aW, aS, aV, aC
    # Shaped like the mask, typed like the state: the mask is uint8.
    return phi_halo.new_empty(mask.shape)


@torch.library.custom_op("defgcm::cg3d_axpy_s", mutates_args=())
def cg3d_axpy_s(
    q: torch.Tensor, s: torch.Tensor, beta: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Return ``(q + beta*s) * mask`` from one launch."""
    reason = axpy_decline_reason(q, s, beta, mask)
    if reason is not None:
        _refuse(reason, "search-direction update")
    return _load_extension().cg3d_axpy_s_fwd(q, s, beta, mask)


@cg3d_axpy_s.register_fake
def _cg3d_axpy_s_fake(q, s, beta, mask):
    del s, beta, mask
    return torch.empty_like(q)


@torch.library.custom_op(
    "defgcm::cg3d_axpy_s_pad", mutates_args=("padded",)
)
def cg3d_axpy_s_pad(
    q: torch.Tensor, beta: torch.Tensor, mask: torch.Tensor,
    padded: torch.Tensor, halo: int,
) -> None:
    """``s <- (q + beta*s) * mask``, in a padded buffer's interior.

    ``s`` is not a separate argument: it IS the interior of ``padded``, which
    this overwrites in place, avoiding a separate full-field copy.

    The border cells are the caller's business
    (``geometry/padded_field.py`` refreshes them); this writes only the
    interior, so a resident buffer keeps the halo it has until then.
    """
    reason = axpy_pad_decline_reason(q, beta, mask, padded, halo)
    if reason is not None:
        _refuse(reason, "padded search-direction update")
    _load_extension().cg3d_axpy_s_pad(q, beta, mask, padded, halo)


@cg3d_axpy_s_pad.register_fake
def _cg3d_axpy_s_pad_fake(q, beta, mask, padded, halo):
    return None


@torch.library.custom_op("defgcm::cg3d_update_xr", mutates_args=())
def cg3d_update_xr(
    x: torch.Tensor,
    s: torch.Tensor,
    r: torch.Tensor,
    qmat: torch.Tensor,
    alpha: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``((x + alpha*s)*mask, (r - alpha*qmat)*mask)`` from one launch."""
    reason = update_decline_reason(x, s, r, qmat, alpha, mask)
    if reason is not None:
        _refuse(reason, "state update")
    x_out, r_out = _load_extension().cg3d_update_xr_fwd(x, s, r, qmat, alpha, mask)
    return x_out, r_out


@cg3d_update_xr.register_fake
def _cg3d_update_xr_fake(x, s, r, qmat, alpha, mask):
    del s, qmat, alpha, mask
    return torch.empty_like(x), torch.empty_like(r)


@torch.library.custom_op("defgcm::cg3d_update_xr_inplace", mutates_args=("x", "r"))
def cg3d_update_xr_inplace(
    x: torch.Tensor, s: torch.Tensor, r: torch.Tensor, qmat: torch.Tensor,
    alpha: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor,
) -> None:
    """Commit an accepted iteration directly into its fixed graph buffers."""
    reason = update_decline_reason(x, s, r, qmat, alpha, mask)
    if reason is not None:
        _refuse(reason, "cg3d_update_xr_inplace")
    _load_extension().cg3d_update_xr_inplace(x, s, r, qmat, alpha, mask, valid)


@cg3d_update_xr_inplace.register_fake
def _cg3d_update_xr_inplace_fake(x, s, r, qmat, alpha, mask, valid):
    return None


@torch.library.custom_op("defgcm::cg3d_thomas_mask", mutates_args=())
def cg3d_thomas_mask(
    zMC: torch.Tensor,
    zML: torch.Tensor,
    zMU: torch.Tensor,
    r: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Vertical-line Thomas apply with the mask folded onto the store."""
    reason = thomas_decline_reason(zMC, zML, zMU, r, mask)
    if reason is not None:
        _refuse(reason, "Thomas apply")
    return _load_extension().cg3d_thomas_mask_fwd(zMC, zML, zMU, r, mask)


@cg3d_thomas_mask.register_fake
def _cg3d_thomas_mask_fake(zMC, zML, zMU, r, mask):
    del zMC, zML, zMU, mask
    return torch.empty_like(r)


# ---------------------------------------------------------------------------
# The dot-folding entry points.
#
# Each returns, alongside the field its non-folding twin returns, the PER-BLOCK
# partial sums of one inner product the graph would otherwise form with a
# multiply kernel and a reduction kernel over the same operands.  The caller
# finishes them with ``torch.sum`` and the same global reduction it used
# before, so the collective and its placement are unchanged.
#

# The summation ORDER differs from ``torch.sum``'s.  The three
# non-folding operators above stay class-A and remain the reference the host
# core and the existing tests use; nothing here replaces them.
#
# The partial arrays are shaped by the launch configuration, which the fake
# implementations cannot know, so they are declared dynamic.  A compiled region
# never sees these ops today (the graph path is captured, not traced), and the
# fakes exist so that a future one gets a meta result rather than a crash.
# ---------------------------------------------------------------------------


@torch.library.custom_op("defgcm::cg3d_thomas_mask_dot", mutates_args=())
def cg3d_thomas_mask_dot(
    zMC: torch.Tensor,
    zML: torch.Tensor,
    zMU: torch.Tensor,
    r: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q, partial sums of <q, r>)`` from one launch."""
    reason = thomas_decline_reason(zMC, zML, zMU, r, mask)
    if reason is not None:
        _refuse(reason, "Thomas apply with folded dot")
    q, partials = _load_extension().cg3d_thomas_mask_dot_fwd(
        zMC, zML, zMU, r, mask
    )
    return q, partials


@cg3d_thomas_mask_dot.register_fake
def _cg3d_thomas_mask_dot_fake(zMC, zML, zMU, r, mask):
    del zMC, zML, zMU, mask
    context = torch.library.get_ctx()
    return torch.empty_like(r), r.new_empty((context.new_dynamic_size(),))


@torch.library.custom_op("defgcm::cg3d_spmv_halo_dot", mutates_args=())
def cg3d_spmv_halo_dot(
    phi_halo: torch.Tensor,
    aW: torch.Tensor,
    aS: torch.Tensor,
    aV: torch.Tensor,
    aC: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(Ap*mask, partial sums of <p, Ap*mask>)`` from one launch.

    ``p`` is the CENTRE of the padded block, which is bitwise the unhaloed
    search direction, so the second operand of the inner product costs no load.
    """
    reason = spmv_decline_reason(phi_halo, aW, aS, aV, aC, mask)
    if reason is not None:
        _refuse(reason, "SPMV with folded dot")
    out, partials = _load_extension().cg3d_spmv_halo_dot_fwd(
        phi_halo, aW, aS, aV, aC, mask
    )
    return out, partials


@cg3d_spmv_halo_dot.register_fake
def _cg3d_spmv_halo_dot_fake(phi_halo, aW, aS, aV, aC, mask):
    del aW, aS, aV, aC
    context = torch.library.get_ctx()
    return (phi_halo.new_empty(mask.shape),
            phi_halo.new_empty((context.new_dynamic_size(),)))


@torch.library.custom_op(
    "defgcm::cg3d_update_xr_inplace_dot", mutates_args=("x", "r")
)
def cg3d_update_xr_inplace_dot(
    x: torch.Tensor, s: torch.Tensor, r: torch.Tensor, qmat: torch.Tensor,
    alpha: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor,
    s_halo: int = 0,
) -> torch.Tensor:
    """Commit an accepted iteration and return partial sums of ``<r, r>``.

    A REJECTED iteration writes neither ``x``/``r`` nor a contribution, so the
    partials are all zero -- and the caller discards the resulting ``err_sq``
    through the same validity ``where`` it already applied.

    ``s_halo`` says what ``s`` is: 0 for a dense field of the state's shape,
    h>0 for a padded buffer ``[nf, nz, ny+2h, nx+2h]`` whose interior holds
    it.  The kernel offsets its index rather than taking a strided view --
    which it could not index, since it walks ``s`` linearly.
    """
    reason = update_decline_reason(x, s, r, qmat, alpha, mask, s_halo=s_halo)
    if reason is not None:
        _refuse(reason, "cg3d_update_xr_inplace_dot")
    return _load_extension().cg3d_update_xr_inplace_dot(
        x, s, r, qmat, alpha, mask, valid, s_halo
    )


@cg3d_update_xr_inplace_dot.register_fake
def _cg3d_update_xr_inplace_dot_fake(x, s, r, qmat, alpha, mask, valid,
                                     s_halo=0):
    del s, qmat, alpha, mask, valid, s_halo
    context = torch.library.get_ctx()
    return x.new_empty((context.new_dynamic_size(),))


__all__ = [
    "DOT_KERNEL_NAMES",
    "KERNEL_NAMES",
    "SUPPORTED_DTYPES",
    "axpy_decline_reason",
    "axpy_pad_decline_reason",
    "bundle_decline_reason",
    "cg3d_axpy_s",
    "cg3d_axpy_s_pad",
    "cg3d_spmv_halo",
    "cg3d_spmv_halo_dot",
    "cg3d_thomas_mask",
    "cg3d_thomas_mask_dot",
    "cg3d_update_xr",
    "cg3d_update_xr_inplace",
    "cg3d_update_xr_inplace_dot",
    "cuda_launch_config",
    "fused_pcg_available",
    "load_extension",
    "mask_binary_reason",
    "mask_reason",
    "operator_shape_reason",
    "spmv_decline_reason",
    "state_reason",
    "thomas_decline_reason",
    "to_kernel_mask",
    "update_decline_reason",
]
