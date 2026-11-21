"""Gate, input adapter and custom-op mount for the fused VIM horizontal phase.

The kernel replaces these stages of
:meth:`VectorInvariantMomentumAdvection.evaluate`: corner curl, the hFacZ corner
mask, both KE gradients, both planetary Coriolis faces, both relative-vorticity
flux faces and the three-term combine, in ONE launch.

Two predicates guard it and both are single-source:

* :func:`config_decline_reason` -- the *adapter's* precondition (grid topology,
  scheme selectors, mask presence).  The kernel has no opinion about these; the
  adapter does, because it is what reproduces the eager halo policy.
* :func:`direct_decline_reason` -- the *kernel's* precondition (device, dtype,
  rank, strides, grad).  The strict custom op and the dispatch shell call the
  same function, so the gate can never accept an input the kernel rejects.

Every ``exchange`` stays outside the kernel, so the halo
semantics -- REPLICATE for KE, ZERO for the velocity/corner halos, distributed
neighbour data when a rank layout is attached -- remain owned by
``locomgcm.geometry.exchange`` alone.

Distributed note: both predicates use shared grad mode as well as local tensor
metadata plus grid topology, which is SPMD-uniform on a homogeneous run.  They
contain no collective, so they cannot wedge.
"""

import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import torch
from locomgcm.accel.tensor_context import grad_mode_requires_reference

from locomgcm.accel import current_rank, dispatch_scope, gate, record_dispatch


_HERE = Path(__file__).resolve().parent
_KEY = "vim_fused_cuda"
_SITE = "VectorInvariantMomentumAdvection.evaluate"
_BACKEND = "cuda"
_SUPPORTED_DTYPES = (torch.float32, torch.float64)

_EXTENSION: Any | None = None
_LOAD_FAILED = False
_COUNTS: Counter = Counter()


def _count(key: str) -> None:
    """Diagnostic counter, advanced only outside the compiled step.

    A module-level Counter mutated inside the traced region is guarded on its
    contents by Dynamo, so the step frame recompiled on every step until the
    limit sent it eager. The ledger
    ignition evidence goes through record_dispatch, which queues under
    compilation; this counter only feeds dispatch_counts().
    """
    if not torch._dynamo.is_compiling():
        _COUNTS[key] += 1
_NATIVE_LAUNCHES = 0


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


def config_decline_reason(
    grid: Any,
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    ke_scheme: str,
    vort_scheme: str,
    split_coriolis: bool,
    cori_scheme: str,
) -> str | None:
    """Boundary precondition: can this call be reproduced bitwise by the core?

    Every branch declined here is one the fused adapter does NOT reproduce --
    the cube leaf, the weighted KE schemes, the 2-D full-metric Coriolis
    variant, the combined-vorticity assembly, the 2-D mask path.  Each stays on
    the frozen eager reference.

    The device/dtype screen lives here too, and deliberately not as a second
    check inside the dispatch shell: this is the ONE predicate the consumer
    boundary consults, so a host-tensor call is refused before the adapter
    allocates a single halo.  The kernel keeps its own contract in
    :func:`direct_decline_reason`, which re-checks the prepared tuple; the two
    are different objects (call vs. prepared tensors), not two copies of one.
    """
    if getattr(grid, "face_count", 1) != 1:
        return "cube_grid"
    if getattr(grid, "cube_corner_indices", ()):
        return "cube_corner_override"
    # Scheme names are those of the module's KE_SCHEMES / VORT_SCHEMES /
    # CORI_SCHEMES; the kernel reproduces selectCoriScheme=0 only.  The
    # 3-D Coriolis term is evaluated in evaluate's shared tail, outside this
    # kernel, so its scheme does not enter here.
    if ke_scheme != "unweighted":
        return "select_ke_scheme"
    if vort_scheme != "enstrophy_hfacz":
        return "select_vort_scheme"
    if not split_coriolis:
        return "combined_coriolis"
    if cori_scheme != "simple":
        return "select_cori_scheme"
    if not grid.metrics.is_uniform_x or not grid.metrics.is_uniform_y:
        return "full_metric_coriolis"
    if not grid.masks.has_3d_masks:
        return "no_3d_masks"
    if u.ndim != 4 or v.ndim != 4:
        return "rank"
    if grid.ny < 2 or grid.nx < 2:
        return "degenerate_extent"
    if u.device.type != "cuda" or v.device.type != "cuda":
        return "non_cuda"
    if u.device != v.device:
        return "device_mismatch"
    if u.dtype not in _SUPPORTED_DTYPES or v.dtype != u.dtype:
        return "dtype"
    if grad_mode_requires_reference():
        return "grad"
    return None


def direct_decline_reason(values: tuple[Any, ...]) -> str | None:
    """Kernel precondition, shared by the strict op and the dispatch shell.

    ``values`` is the positional prepared-input tuple produced by
    :func:`prepare_inputs` minus its two trailing booleans; ``None`` entries
    are the two optional edge masks.
    """
    tensors = [value for value in values if value is not None]
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        return "non_tensor"
    if any(value.device.type != "cuda" for value in tensors):
        return "non_cuda"
    reference = tensors[0]
    if any(value.device != reference.device for value in tensors):
        return "device_mismatch"
    if reference.dtype not in _SUPPORTED_DTYPES:
        return "dtype"
    for value in tensors:
        if value.dtype is torch.bool:
            continue
        if value.dtype != reference.dtype:
            return "dtype"
    if any(stride < 0 for value in tensors for stride in value.stride()):
        return "negative_stride"
    if any(value.numel() == 0 for value in tensors):
        return "empty_shape"
    if grad_mode_requires_reference():
        return "grad"
    return None


# ---------------------------------------------------------------------------
# Input adapter
# ---------------------------------------------------------------------------


def prepare_inputs(
    u: torch.Tensor,
    v: torch.Tensor,
    grid: Any,
    *,
    coriolis_parameter: float,
    coriolis_profile_g: tuple[float, ...] | None,
    omega: float,
    coriolis_profile_tensor: torch.Tensor | None = None,
    metrics: Any,
) -> tuple[Any, ...]:
    """Build the kernel's pre-haloed inputs using the eager halo policy.

    The exchange calls below are the ONLY duplication of eager code in this
    package.  Each carries the source line it mirrors.
    """
    from locomgcm.geometry.exchange import FillSpec, exchange
    from locomgcm.operators import c_grid
    from locomgcm.dynamics._vector_invariant_topology import (
        corner_coriolis_field_for_grid,
    )
    from locomgcm.dynamics.kinetic_energy import ke_center_unweighted
    from locomgcm.dynamics.vector_invariant_momentum import (
        _REGION_X_HALO,
        _REGION_Y_HALO,
        _compute_hFacZ,
        _edge_pair_mask,
        _region_u_row_metrics,
        _region_u_y_halo,
        _region_v_dyg,
        _region_v_x_halo,
    )

    dtype, device = u.dtype, u.device
    ny, nx = grid.ny, grid.nx

    # config_decline_reason admits only the unweighted KE scheme.
    ke = ke_center_unweighted(u, v)
    # vector_invariant_momentum.py:422-433 / :482-493 (fill=REPLICATE).
    ke_x_ext = exchange(
        ke, grid=grid, location="U", extent=_REGION_X_HALO,
        fill=FillSpec.REPLICATE,
    )
    ke_y_ext = exchange(
        ke, grid=grid, location="V", extent=_REGION_Y_HALO,
        fill=FillSpec.REPLICATE,
    )

    # c_grid.py:837-851 -- the curl's own corner halos.
    u_corner = exchange(
        u, grid=grid, location="F", extent=c_grid._HALO_XY, fill=FillSpec.ZERO,
    )
    v_corner = exchange(
        v, grid=grid, location="F", extent=c_grid._HALO_XY, fill=FillSpec.ZERO,
    )

    # Shared velocity halos -- reused verbatim, not re-derived.
    v_x_halo = _region_v_x_halo(v, grid)
    u_y_halo = _region_u_y_halo(u, grid)

    hfacz = _compute_hFacZ(grid, dtype=dtype, device=device)
    # ``corner_coriolis_field`` resolves ``coriolis_profile_g`` through a
    # ``constant_cache`` key that starts with ``id(profile)``, which is
    # ``id() with unsupported args (TupleVariable(length=ny+1))`` -- a graph
    # break inside the traced step.  The profile tensor is therefore the
    # eager pre-step's answer for this (dtype, device); the eager block reads it
    # from the same memo (``VectorInvariantMomentumAdvection._coriolis_profile_tensor``),
    # so both consumers get the identical tensor.  ``None`` keeps the legacy
    # in-place resolution for direct eager callers.
    fg = corner_coriolis_field_for_grid(
        u,
        grid,
        coriolis_parameter=coriolis_parameter,
        coriolis_profile_g=coriolis_profile_g,
        omega=omega,
        profile_tensor=coriolis_profile_tensor,
    )

    dxC = grid.metrics.dxC_for(dtype, device)
    # c_grid.py:876-882 -- haloed dxC row block; rows 1..ny are dxC itself, so
    # this one tensor serves BOTH the curl and the u-face KE gradient.
    dxC_ext = exchange(
        dxC, grid=grid, location="F", extent=c_grid._HALO_Y, fill=FillSpec.ZERO,
    )
    dyC = grid.metrics.dyC_for(dtype, device)
    rAz = grid.metrics.rAz_for(dtype, device)

    dxG_south, dxG_north, recip_dxC_u = _region_u_row_metrics(metrics)
    dyG, recip_dyC = _region_v_dyg(metrics)

    mask_w = grid.masks.u_float(dtype=dtype, device=device)
    mask_s = grid.masks.v_float(dtype=dtype, device=device)

    # c_grid.py:853-874 -- corner support masks.
    x_core = torch.ones(nx, dtype=torch.bool, device=device)
    x_core_ext = exchange(
        x_core, grid=grid, location="F", extent=c_grid._HALO_X,
        fill=FillSpec.ZERO,
    )
    x_corner_mask = x_core_ext[:-1] & x_core_ext[1:]
    y_core = torch.ones(ny, dtype=torch.bool, device=device)
    y_core_ext = exchange(
        y_core, grid=grid, location="F", extent=c_grid._HALO_Y,
        fill=FillSpec.ZERO,
    )
    y_corner_mask = y_core_ext[:-1] & y_core_ext[1:]

    edge_x = _edge_pair_mask(
        nx, grid=grid, axis="x", extent=_REGION_X_HALO, device=device,
    )
    edge_y = _edge_pair_mask(
        ny, grid=grid, axis="y", extent=_REGION_Y_HALO, device=device,
    )
    # ``_edge_pair_mask`` returns None exactly when ``edge_face_couples`` is
    # True (vector_invariant_momentum.py:1191-1192), so the eager
    # ``elif not edge_face_couples: _zero_*_edge_faces`` branch at :1067/:1150
    # is unreachable from these call sites and the kernel implements only the
    # mask form.  Assert the equivalence rather than duplicate a dead branch.
    if (edge_x is None) != grid.edge_face_couples("x") or (
        edge_y is None
    ) != grid.edge_face_couples("y"):
        raise RuntimeError(
            "VIM fused: edge_pair_mask presence no longer mirrors "
            "edge_face_couples; the eager zero-edge branch would be live"
        )

    # c_grid.py:1042-1046 -- the periodic-seam copy, expressed as a source
    # index remap inside the kernel.
    periodic_x_copy = bool(
        grid.boundary_topology.periodic_x
        and not c_grid._axis_is_decomposed(grid, "x")
    )
    periodic_y_copy = bool(
        grid.boundary_topology.periodic_y
        and not c_grid._axis_is_decomposed(grid, "y")
    )

    return (
        ke_x_ext,
        ke_y_ext,
        u_corner,
        v_corner,
        v_x_halo,
        u_y_halo,
        hfacz,
        fg,
        dxC_ext,
        dyC,
        rAz,
        dxG_south,
        dxG_north,
        recip_dxC_u,
        dyG,
        recip_dyC,
        mask_w,
        mask_s,
        x_corner_mask,
        y_corner_mask,
        edge_x,
        edge_y,
        periodic_x_copy,
        periodic_y_copy,
    )


# ---------------------------------------------------------------------------
# Extension load
# ---------------------------------------------------------------------------


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
        raise RuntimeError("VIM fused extension previously failed to load")
    from torch.utils.cpp_extension import load

    cxx = _find_conda_cxx()
    if cxx is not None:
        os.environ.setdefault("CXX", cxx)
        os.environ.setdefault("CUDAHOSTCXX", cxx)
    # -fmad=false keeps the transcription bitwise against eager.
    cuda_flags = ["-O3", "-prec-div=true", "-prec-sqrt=true", "-fmad=false"]
    if cxx is not None:
        cuda_flags.append(f"-ccbin={cxx}")
    try:
        _EXTENSION = load(
            name="locomgcm_vim_fused",
            sources=[str(_HERE / "extension.cpp"), str(_HERE / "kernel.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=cuda_flags,
            with_cuda=True,
            verbose=os.environ.get("LOCOMGCM_VIM_FUSED_VERBOSE", "0") == "1",
        )
    except Exception:
        _LOAD_FAILED = True
        raise
    return _EXTENSION


# ---------------------------------------------------------------------------
# Custom op
# ---------------------------------------------------------------------------


@torch.library.custom_op("locomgcm::vim_fused", mutates_args=())
def vim_fused(
    ke_x_ext: torch.Tensor,
    ke_y_ext: torch.Tensor,
    u_corner: torch.Tensor,
    v_corner: torch.Tensor,
    v_x_halo: torch.Tensor,
    u_y_halo: torch.Tensor,
    hfacz: torch.Tensor,
    fg: torch.Tensor,
    dxC_ext: torch.Tensor,
    dyC: torch.Tensor,
    rAz: torch.Tensor,
    dxG_south: torch.Tensor,
    dxG_north: torch.Tensor,
    recip_dxC_u: torch.Tensor,
    dyG: torch.Tensor,
    recip_dyC: torch.Tensor,
    mask_w: torch.Tensor,
    mask_s: torch.Tensor,
    x_corner_mask: torch.Tensor,
    y_corner_mask: torch.Tensor,
    edge_pair_mask_x: Optional[torch.Tensor],
    edge_pair_mask_y: Optional[torch.Tensor],
    periodic_x_copy: bool,
    periodic_y_copy: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute one strict CUDA launch; no fallback and no autograd."""
    global _NATIVE_LAUNCHES
    values = (
        ke_x_ext, ke_y_ext, u_corner, v_corner, v_x_halo, u_y_halo, hfacz, fg,
        dxC_ext, dyC, rAz, dxG_south, dxG_north, recip_dxC_u, dyG, recip_dyC,
        mask_w, mask_s, x_corner_mask, y_corner_mask, edge_pair_mask_x,
        edge_pair_mask_y,
    )
    reason = direct_decline_reason(values)
    if reason is not None:
        raise RuntimeError(f"VIM fused custom op is inapplicable: {reason}")
    result = load_extension().vim_fused_fwd(
        ke_x_ext, ke_y_ext, u_corner, v_corner, v_x_halo, u_y_halo, hfacz, fg,
        dxC_ext, dyC, rAz, dxG_south, dxG_north, recip_dxC_u, dyG, recip_dyC,
        mask_w, mask_s, x_corner_mask, y_corner_mask, edge_pair_mask_x,
        edge_pair_mask_y, bool(periodic_x_copy), bool(periodic_y_copy),
    )
    _NATIVE_LAUNCHES += 1
    return result[0], result[1]


@vim_fused.register_fake
def _vim_fused_fake(
    ke_x_ext: torch.Tensor,
    ke_y_ext: torch.Tensor,
    u_corner: torch.Tensor,
    v_corner: torch.Tensor,
    v_x_halo: torch.Tensor,
    u_y_halo: torch.Tensor,
    hfacz: torch.Tensor,
    fg: torch.Tensor,
    dxC_ext: torch.Tensor,
    dyC: torch.Tensor,
    rAz: torch.Tensor,
    dxG_south: torch.Tensor,
    dxG_north: torch.Tensor,
    recip_dxC_u: torch.Tensor,
    dyG: torch.Tensor,
    recip_dyC: torch.Tensor,
    mask_w: torch.Tensor,
    mask_s: torch.Tensor,
    x_corner_mask: torch.Tensor,
    y_corner_mask: torch.Tensor,
    edge_pair_mask_x: Optional[torch.Tensor],
    edge_pair_mask_y: Optional[torch.Tensor],
    periodic_x_copy: bool,
    periodic_y_copy: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        u_corner, v_corner, v_x_halo, u_y_halo, hfacz, fg, dxC_ext, dyC, rAz,
        dxG_south, dxG_north, recip_dxC_u, dyG, recip_dyC, mask_w, mask_s,
        x_corner_mask, y_corner_mask, edge_pair_mask_x, edge_pair_mask_y,
        periodic_x_copy, periodic_y_copy,
    )
    batch, levels, ny, nx_plus_two = ke_x_ext.shape
    nx = nx_plus_two - 2
    del ke_y_ext
    return (
        ke_x_ext.new_empty((batch, levels, ny, nx + 1)),
        ke_x_ext.new_empty((batch, levels, ny + 1, nx)),
    )


# ---------------------------------------------------------------------------
# Counting dispatch shell
# ---------------------------------------------------------------------------


def _record(outcome: str, *, reason: str | None = None) -> None:
    record_dispatch(
        _KEY,
        outcome=outcome,
        variant="split_coriolis",
        site=_SITE,
        backend=_BACKEND,
        rank=current_rank(),
        reason=reason,
    )


def vim_fused_dispatch(
    u: torch.Tensor,
    v: torch.Tensor,
    grid: Any,
    *,
    ke_scheme: str,
    vort_scheme: str,
    split_coriolis: bool,
    cori_scheme: str,
    coriolis_parameter: float,
    coriolis_profile_g: tuple[float, ...] | None,
    omega: float,
    coriolis_profile_tensor: torch.Tensor | None = None,
    metrics: Any,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return the fused ``(tend_u, tend_v)``, or ``None`` to run eager.

    ``None`` is the contract with the consumer boundary: the caller keeps the
    frozen eager block, which is therefore the only fallback implementation.
    """
    if not gate(_KEY).effective_value:
        _count("disabled")
        return None

    reason = config_decline_reason(
        grid,
        u,
        v,
        ke_scheme=ke_scheme,
        vort_scheme=vort_scheme,
        split_coriolis=split_coriolis,
        cori_scheme=cori_scheme,
    )
    if reason is not None:
        _count(f"fallback:{reason}")
        _record("inapplicable", reason=reason)
        return None

    with dispatch_scope(_KEY):
        prepared = prepare_inputs(
            u,
            v,
            grid,
            coriolis_parameter=coriolis_parameter,
            coriolis_profile_g=coriolis_profile_g,
            omega=omega,
            coriolis_profile_tensor=coriolis_profile_tensor,
            metrics=metrics,
        )
        reason = direct_decline_reason(prepared[:-2])
        if reason is not None:
            _count(f"fallback:{reason}")
            _record("inapplicable", reason=reason)
            return None
        try:
            tend_u, tend_v = vim_fused(*prepared)
        except Exception as error:
            _count("error")
            _record("error", reason=type(error).__name__)
            raise
    _count("requested-hit")
    _record("hit")
    return tend_u, tend_v


def dispatch_counts() -> dict[str, int]:
    return dict(_COUNTS)


def reset_dispatch_counts() -> None:
    _COUNTS.clear()


def native_launch_count() -> int:
    """Extension-side launch counter; 0 with the gate off is the null control."""
    return _NATIVE_LAUNCHES


def reset_native_launch_count() -> None:
    global _NATIVE_LAUNCHES
    _NATIVE_LAUNCHES = 0


__all__ = [
    "config_decline_reason",
    "direct_decline_reason",
    "dispatch_counts",
    "load_extension",
    "native_launch_count",
    "prepare_inputs",
    "reset_dispatch_counts",
    "reset_native_launch_count",
    "vim_fused",
    "vim_fused_dispatch",
]
