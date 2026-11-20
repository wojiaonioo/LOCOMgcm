"""Strict lazy loader for vim coriolis u.

The direct binding accepts 8 broadcast-compatible 0-dimensional or 1-dimensional or 2-dimensional or 3-dimensional or 4-dimensional float64
CUDA tensors with arbitrary non-negative strides.  It never inserts contiguous
copies, never falls back, and has no autograd implementation.  Gate selection and
the eager fallback live one level up, in ``dispatcher.py``, which shares this
module's :func:`direct_decline_reason` predicate so the two can never disagree
about what the kernel supports.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from defgcm.accel.tensor_context import grad_mode_requires_reference


_HERE = Path(__file__).resolve().parent
_KEY = "vim_coriolis_u"
_EXTENSION_NAME = "defgcm_vim_coriolis_u_cuda"
#: The dtypes kernel.cu instantiates.
_SUPPORTED_DTYPES = (torch.float64,)
_EXTENSION: Any | None = None
_LOAD_FAILED = False
# Precise division and separate multiply/add rounding are numerical
# requirements, not optimisation preferences: -fmad=false is what makes this
# kernel's result reproduce the eager reference bit for bit.
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
        raise RuntimeError("vim coriolis u CUDA extension previously failed to load")
    from torch.utils.cpp_extension import load

    try:
        _EXTENSION = load(
            name=_EXTENSION_NAME,
            sources=[str(_HERE / "extension.cpp"), str(_HERE / "kernel.cu")],
            extra_cflags=_CPP_FLAGS,
            extra_cuda_cflags=_cuda_flags(),
            with_cuda=True,
            verbose=os.environ.get("DEFGCM_VIM_CORIOLIS_U_CUDA_VERBOSE", "0") == "1",
        )
    except Exception:
        _LOAD_FAILED = True
        raise
    _push_launch_decision()
    return _EXTENSION


def _push_launch_decision() -> None:
    if not torch.cuda.is_available():
        return
    from defgcm.accel import record_launch_decision

    device = torch.cuda.current_device()
    with torch.cuda.device(device):
        config = dict(_EXTENSION.launch_config(device))
    record_launch_decision(
        kernel=_KEY, device=str(torch.device("cuda", device)), decision=config
    )


def compile_vim_coriolis_u_extension() -> Any:
    """Compile/load the extension now, raising the complete build failure."""
    return _load_extension()


def vim_coriolis_u_launch_config(device: int | None = None) -> dict[str, Any]:
    """Return the native occupancy decision cached for this kernel and device."""
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("vim coriolis u launch configuration requires CUDA")
        device = torch.cuda.current_device()
    with torch.cuda.device(device):
        return dict(_load_extension().launch_config(device))


def vim_coriolis_u_launch_count() -> int:
    """Return the native extension's successful launch-request count."""
    return int(_load_extension().native_launch_count())


def reset_vim_coriolis_u_launch_count() -> None:
    """Reset the native extension launch-request counter."""
    _load_extension().reset_native_launch_count()


#: One reason token per applicability rule.  The strict entry point raises on
#: these; the production dispatcher records them as ``inapplicable`` and falls
#: back.  Both read the SAME predicate below, so the gate can never accept an
#: input the kernel would reject, nor decline one it would have served.  Two
#: copies of this logic drift into either a crash or a silent no-op.
_DECLINE_MESSAGES = {
    "arity": "native vim_coriolis_u dispatch requires exactly 8 tensors",
    "non_tensor": "native vim_coriolis_u dispatch requires tensor inputs",
    "layout": "native vim_coriolis_u inputs must use strided layout",
    "non_cuda": "native vim_coriolis_u dispatch requires CUDA inputs",
    "device_mismatch": "native vim_coriolis_u inputs must be on one CUDA device",
    "dtype": "native vim_coriolis_u dispatch requires float64 inputs, all of one dtype",
    "ndim": "native vim_coriolis_u inputs must have ndim in [0, 1, 2, 3, 4]",
    "trailing_shape": "native vim_coriolis_u trailing input dimensions must match",
    "empty_shape": "native vim_coriolis_u inputs must have positive extents",
    "batch_broadcast": "native vim_coriolis_u leading batch dimensions are not broadcast-compatible",
    "shape": "native vim_coriolis_u input shapes do not satisfy the declared stencil pattern",
    "negative_stride": "native vim_coriolis_u inputs must not have negative strides",
    "edge_face_couples": "native vim_coriolis_u edge_face_couples must be bool",
    "autograd": "native vim_coriolis_u dispatch has no autograd implementation",
}


#: Declared input shapes. Each entry is one pattern, or several of which the
#: input must match one -- an input whose RANK depends on the dispatch variant
#: needs the latter. A term is "*" (any extent, and the dimension may be absent),
#: a bare symbol, a symbol with an integer offset, or a literal extent.
_SHAPE_PATTERNS: dict[str, tuple] = {'v_ext': ('*', 'nz', 'ny+1', 'nx+2'), 'corner_value': (('ny+1', 'nx+1'), ('*', 'nz', 'ny+1', 'nx+1')), 'dxG_south': ('ny', '1'), 'dxG_north': ('ny', '1'), 'recip_dxC_u': ('ny', '1'), 'hFacZ': ('nz', 'ny+1', 'nx+1'), 'face_mask': ('nz', 'ny', 'nx+1'), 'edge_pair_mask': ('nx+1',)}


def _shape_terms(pattern: tuple) -> list[tuple]:
    """(literal, symbol, offset, wildcard) per dimension."""
    out = []
    for text in pattern:
        text = str(text).strip()
        if text == "*":
            out.append((None, None, 0, True))
        elif text.lstrip("-").isdigit():
            out.append((int(text), None, 0, False))
        else:
            symbol, offset = text, 0
            for sign in ("+", "-"):
                if sign in text:
                    symbol, _, tail = text.partition(sign)
                    offset = int(tail) * (-1 if sign == "-" else 1)
                    break
            out.append((None, symbol.strip(), offset, False))
    return out


def _align(shape: tuple, terms: list[tuple]):
    wild = sum(1 for t in terms if t[3])
    fixed = terms[wild:]
    if len(shape) < len(fixed) or len(shape) > len(terms):
        return None
    return list(zip(fixed, shape[len(shape) - len(fixed):]))


#: ``_SHAPE_PATTERNS`` parsed once. Both the shape pass and the batch pass read
#: this one copy, so they can never disagree about how a pattern aligns.
_PARSED_PATTERNS: dict[str, list[list[tuple]]] = {
    name: [
        _shape_terms(alt)
        for alt in (pat if pat and isinstance(pat[0], tuple) else (pat,))
    ]
    for name, pat in _SHAPE_PATTERNS.items()
}


def _fit(shape: tuple, alternatives):
    """The first alternative this shape aligns against, or None."""
    for terms in alternatives:
        pairs = _align(shape, terms)
        if pairs is not None:
            return pairs
    return None


def _batch_broadcast_conflict(present) -> bool:
    """True when the leading extents cannot share one batch index.

    A leading ``"*"`` term is unconstrained BY the patterns -- ``_align`` never
    pairs it and ``_check_shapes`` skips it -- but it is not unconstrained by
    the kernel.  ``extension.cpp`` sizes the output from the LARGER leading
    extent it sees, and ``kernel.cu``'s ``broadcast_strides4`` zeroes an input's
    batch stride only when that input's own extent is 1.  So an extent of 1
    broadcasts and equal extents agree, but two different non-unit extents make
    the kernel walk the shorter input past its end -- an out-of-range global
    read on input the shape pass alone accepts.
    """
    extents = set()
    for name, value in present:
        alternatives = _PARSED_PATTERNS.get(name)
        if alternatives is None:
            continue
        shape = tuple(value.shape)
        pairs = _fit(shape, alternatives)
        if pairs is None:
            continue
        for extent in shape[: len(shape) - len(pairs)]:
            if extent != 1:
                extents.add(extent)
    return len(extents) > 1


def _check_shapes(present) -> str | None:
    """Return the first shape violation, or None. Two passes: bind, then verify."""
    bound: dict[str, int] = {}
    parsed = _PARSED_PATTERNS
    fit = _fit

    for name, value in present:
        alternatives = parsed.get(name)
        if alternatives is None:
            continue
        pairs = fit(tuple(value.shape), alternatives)
        if pairs is None:
            continue
        for (literal, symbol, offset, wild), extent in pairs:
            if symbol and offset == 0 and symbol not in bound:
                bound[symbol] = extent

    for name, value in present:
        alternatives = parsed.get(name)
        if alternatives is None:
            continue
        shape = tuple(value.shape)
        pairs = fit(shape, alternatives)
        if pairs is None:
            return f"{name}: shape {list(shape)} fits no declared pattern"
        for (literal, symbol, offset, wild), extent in pairs:
            if wild:
                continue
            if literal is not None:
                if extent != literal:
                    return f"{name}: expected {literal}, got {extent}"
                continue
            if symbol not in bound:
                bound[symbol] = extent - offset
            want = bound[symbol] + offset
            if extent != want:
                return f"{name}: expected {symbol}{offset:+d}={want}, got {extent}"
    return None



def direct_decline_reason(
    values: tuple[torch.Tensor, ...], *, edge_face_couples: bool
) -> str | None:
    """Return the first unmet applicability rule, or ``None`` when supported."""
    if len(values) != 8:
        return "arity"
    names = ('v_ext', 'corner_value', 'dxG_south', 'dxG_north', 'recip_dxC_u', 'hFacZ', 'face_mask', 'edge_pair_mask')
    optional = ('hFacZ', 'face_mask', 'edge_pair_mask')
    # Optional inputs arrive as None on the variants that omit them;
    # every tensor rule below runs over the present ones only.
    present = [
        (name, value)
        for name, value in zip(names, values)
        if not (value is None and name in optional)
    ]
    if any(not isinstance(value, torch.Tensor) for _, value in present):
        return "non_tensor"
    reference = present[0][1]
    if any(value.layout != torch.strided for _, value in present):
        return "layout"
    if any(value.device.type != "cuda" for _, value in present):
        return "non_cuda"
    if any(value.device != reference.device for _, value in present):
        return "device_mismatch"
    # Inputs with a declared dtype of their own -- a bool mask beside
    # float data -- are checked against that dtype, not against the
    # kernel's floating dtype, and are excluded from the uniform rule.
    typed = {'edge_pair_mask': 'bool'}
    for name, value in present:
        want = typed.get(name)
        if want is not None and value.dtype != getattr(torch, want):
            return "dtype"
    floating = [(n, v) for n, v in present if n not in typed]
    reference = floating[0][1]
    # Mirrors the dtype instantiation in kernel.cu.
    if reference.dtype not in _SUPPORTED_DTYPES:
        return "dtype"
    if any(value.dtype != reference.dtype for _, value in floating):
        return "dtype"
    if any(value.ndim not in (0, 1, 2, 3, 4) for _, value in present):
        return "ndim"
    # Shape patterns supersede the uniform trailing-dims rule: a
    # stencil reads offset neighbourhoods, so its inputs differ from
    # each other by construction and equality would decline them all.
    if _check_shapes(present) is not None:
        return "shape"
    # Runs only once every shape fits a pattern -- before that the leading
    # extents are not identified, so the question is not yet well posed.
    if _batch_broadcast_conflict(present):
        return "batch_broadcast"
    if any(value.numel() == 0 for _, value in present):
        return "empty_shape"
    if any(stride < 0 for _, value in present for stride in value.stride()):
        return "negative_stride"
    if not isinstance(edge_face_couples, bool):
        return "edge_face_couples"
    # A rank's first halo or a prescribed face can be locally inactive even
    # in a differentiated run. Select this inference-only kernel from the
    # shared grad context, never from rank-local requires_grad flags.
    if grad_mode_requires_reference():
        return "autograd"
    return None


def _check_direct(values: tuple[torch.Tensor, ...], *, edge_face_couples: bool) -> None:
    reason = direct_decline_reason(values, edge_face_couples=edge_face_couples)
    if reason is not None:
        raise RuntimeError(_DECLINE_MESSAGES[reason])


def u_face_coriolis_from_haloed_v_row_metrics_cuda_direct(
    v_ext: torch.Tensor,
    corner_value: torch.Tensor,
    *,
    dxG_south: torch.Tensor,
    dxG_north: torch.Tensor,
    recip_dxC_u: torch.Tensor,
    hFacZ: torch.Tensor | None = None,
    face_mask: torch.Tensor | None = None,
    edge_pair_mask: torch.Tensor | None = None,
    edge_face_couples: bool = True,
) -> torch.Tensor:
    """Return the result from exactly 1 native CUDA launch(es)."""
    values = (
        v_ext,
        corner_value,
        dxG_south,
        dxG_north,
        recip_dxC_u,
        hFacZ,
        face_mask,
        edge_pair_mask,
    )
    _check_direct(values, edge_face_couples=edge_face_couples)
    return vim_coriolis_u_fwd(*values, edge_face_couples)


@torch.library.custom_op("defgcm::vim_coriolis_u_fwd", mutates_args=())
def vim_coriolis_u_fwd(
    v_ext: torch.Tensor,
    corner_value: torch.Tensor,
    dxG_south: torch.Tensor,
    dxG_north: torch.Tensor,
    recip_dxC_u: torch.Tensor,
    hFacZ: torch.Tensor | None,
    face_mask: torch.Tensor | None,
    edge_pair_mask: torch.Tensor | None,
    edge_face_couples: bool,
) -> torch.Tensor:
    values = (
        v_ext, corner_value, dxG_south, dxG_north, recip_dxC_u,
        hFacZ, face_mask, edge_pair_mask,
    )
    return _load_extension().vim_coriolis_u_fwd(*values, edge_face_couples)


@vim_coriolis_u_fwd.register_fake
def _vim_coriolis_u_fwd_fake(
    v_ext, corner_value, dxG_south, dxG_north, recip_dxC_u,
    hFacZ, face_mask, edge_pair_mask, edge_face_couples,
):
    shape = list(v_ext.shape)
    if corner_value.ndim == 4:
        shape[0] = max(shape[0], corner_value.shape[0])
    shape[-2] = recip_dxC_u.shape[-2]
    shape[-1] = corner_value.shape[-1]
    return v_ext.new_empty(shape)


__all__ = [
    "compile_vim_coriolis_u_extension",
    "direct_decline_reason",
    "vim_coriolis_u_launch_config",
    "vim_coriolis_u_launch_count",
    "u_face_coriolis_from_haloed_v_row_metrics_cuda_direct",
    "reset_vim_coriolis_u_launch_count",
]
