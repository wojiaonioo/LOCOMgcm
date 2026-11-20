"""Gate, input adapter and custom-op mount for the fused Leith/Smag phase.

Two CUDA launches when every domain edge is a wall or open boundary, or three
when any edge couples (a periodic axis or a rank seam), replace these stages of
:meth:`HorizontalViscosity.evaluate`: horizontal divergence, the
corner curl pair, the dry-corner side mask, the tension, the Leith vorticity
gradients and both point measures, the Smag strain magnitudes, the harmonic
coefficient assembly, the base viscosity and the div-vort dissipation stencil.

Each split is fixed by an eager exchange: distributed coefficient assembly
adds the three measure exchanges, while the div-vort split retains the
all-faces ``hDiv_flux`` exchanges (``HorizontalViscosity._divvort_stencil``).
Every ``exchange`` stays outside the kernels.

Two predicates guard it and both are single-source:

* :func:`config_decline_reason` -- the *adapter's* precondition (grid topology,
  scheme selectors, switch combinations, mask presence).  The kernels have no
  opinion about these; the adapter does, because it is what reproduces the
  eager halo and memo policy.
* :func:`direct_decline_reason` -- the *kernels'* precondition (device, dtype,
  strides, grad).  The strict custom ops and the dispatch shell call the same
  function, so the gate can never accept an input the kernels reject.

Grids with a coupled edge (distributed layouts and single-tile periodic axes,
``_horizontal_diffusion_topology._any_edge_couples``) use two additional kernels around
the eager exchange sites:
P materialises vorticity gradients and tension, the adapter exchanges them, and
Q assembles the coefficients consumed by the unchanged div-vort stage.  Both
predicates read local tensor metadata, grid topology and the shared grad mode,
which are SPMD-uniform on a homogeneous run; they contain no collective.
"""

import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import torch
from defgcm.accel.tensor_context import grad_mode_requires_reference

from defgcm.accel import current_rank, dispatch_scope, gate, record_dispatch


_HERE = Path(__file__).resolve().parent
_KEY = "hvisc_fused_cuda"
_SITE = "HorizontalViscosity.evaluate"
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
    module: Any,
    grid: Any,
    u: torch.Tensor,
    v: torch.Tensor,
) -> str | None:
    """Boundary precondition: can this call be reproduced bitwise by the cores?

    Every branch declined here is one the fused adapter does NOT reproduce --
    the cube leaf, the strain/divvort schemes, no-slip side drag, the
    divergence-Leith and full-Leith assemblies, the biharmonic chain, the
    grid floor/cap, the unit-hFac fall-back and the non-2-D (cube) metric
    layouts.  Each stays on the frozen eager reference.

    The device/dtype screen lives here too, ordered LAST on purpose: every rule
    above it is device-independent, so a CPU host can assert that the shipped
    production layout reaches the kernel and is refused only for its device
    (a reachability negative control).
    """
    if getattr(grid, "face_count", 1) != 1:
        return "cube_grid"
    if getattr(module, "viscosity_scheme", None) != "leith_smag":
        return "viscosity_scheme"
    if module.no_slip_sides:
        return "no_slip_sides"
    if module._leith_div_active():
        return "leith_div"
    if module._biharmonic_active():
        return "biharmonic"
    if float(module.visc_ah_grid_min) > 0.0 or float(module.visc_ah_grid_max) > 0.0:
        return "grid_visc_floor_cap"
    if grid.masks.hFacC is None:
        return "no_hfac"
    if u.ndim != 4 or v.ndim != 4:
        return "rank"
    if grid.ny < 2 or grid.nx < 2:
        return "degenerate_extent"

    dtype, device = u.dtype, u.device
    metrics = grid.metrics
    for name in ("dxC", "dxG", "dyC", "dyG", "dxF", "dyF", "dxV", "dyU", "rAz"):
        if getattr(metrics, f"{name}_for")(dtype, device).ndim != 2:
            return "metric_rank"

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
    """Kernel precondition, shared by the strict ops and the dispatch shell.

    ``values`` is a positional prepared-input tuple with the trailing
    non-tensor arguments removed; ``None`` entries are the optional masks and
    halo tensors.
    """
    tensors = [value for value in values if value is not None]
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        return "non_tensor"
    if not tensors:
        return "empty_shape"
    if any(value.device.type != "cuda" for value in tensors):
        return "non_cuda"
    reference = tensors[0]
    if any(value.device != reference.device for value in tensors):
        return "device_mismatch"
    if reference.dtype not in _SUPPORTED_DTYPES:
        return "dtype"
    if any(value.dtype != reference.dtype for value in tensors):
        return "dtype"
    if any(stride < 0 for value in tensors for stride in value.stride()):
        return "negative_stride"
    if any(value.numel() == 0 for value in tensors):
        return "empty_shape"
    if grad_mode_requires_reference():
        return "grad"
    return None


def _contract_check(condition: bool, message: str) -> None:
    torch._check(condition, lambda: f"hvisc fused multi-rank contract: {message}")


def _check_contract_tensor(
    value: torch.Tensor,
    name: str,
    reference: torch.Tensor,
) -> None:
    _contract_check(value.device.type == "cuda", f"{name} must be a CUDA tensor")
    _contract_check(value.device == reference.device, f"{name} must share one CUDA device")
    _contract_check(value.dtype == reference.dtype, f"{name} must share the reference dtype")


def _check_contract_shape(
    value: torch.Tensor,
    name: str,
    reference: torch.Tensor,
    shape: tuple[Any, ...],
) -> None:
    _check_contract_tensor(value, name, reference)
    _contract_check(value.dim() == len(shape), f"{name} must be {len(shape)}-dimensional")
    if value.dim() == len(shape):
        for axis, extent in enumerate(shape):
            _contract_check(value.size(axis) == extent, f"{name} has an unexpected shape")


def _check_contract_metric(
    value: torch.Tensor,
    name: str,
    reference: torch.Tensor,
    shape: tuple[Any, Any],
) -> None:
    _check_contract_tensor(value, name, reference)
    _contract_check(value.dim() >= 2, f"{name} must have at least two dimensions")
    if value.dim() >= 2:
        for axis in range(value.dim() - 2):
            _contract_check(value.size(axis) == 1, f"{name} may only carry singleton leading dimensions")
        _contract_check(value.size(-2) == shape[0], f"{name} has an unexpected shape")
        _contract_check(value.size(-1) == shape[1], f"{name} has an unexpected shape")


def _validate_mr_pre_contract(values: tuple[Any, ...]) -> None:
    (
        u, v, u_ext, v_dyC_ext, dxC_ext, dxG, dyG, dyF, recip_rA, recip_rAz,
        hFacW, hFacS, recip_hFacC, maskW, maskS, hfacz, maskInC, maskInW, maskInS,
    ) = values
    _contract_check(u.dtype in _SUPPORTED_DTYPES, "inputs must be float32 or float64")
    _contract_check(u.dim() == 4, "u must be four-dimensional")
    if u.dim() != 4:
        return
    batch, levels, ny, nx1 = u.shape
    nx = nx1 - 1
    _contract_check(nx >= 2, "requires ny >= 2 and nx >= 2")
    _contract_check(ny >= 2, "requires ny >= 2 and nx >= 2")
    _contract_check(batch > 0, "extents must be positive")
    _contract_check(levels > 0, "extents must be positive")
    for value, name, shape in (
        (u, "u", (batch, levels, ny, nx + 1)),
        (v, "v", (batch, levels, ny + 1, nx)),
        (u_ext, "u_ext", (batch, levels, ny + 2, nx + 1)),
        (v_dyC_ext, "v_dyC_ext", (batch, levels, ny + 1, nx + 2)),
        (hFacW, "hFacW", (levels, ny, nx + 1)),
        (hFacS, "hFacS", (levels, ny + 1, nx)),
        (recip_hFacC, "recip_hFacC", (levels, ny, nx)),
        (maskW, "maskW", (levels, ny, nx + 1)),
        (maskS, "maskS", (levels, ny + 1, nx)),
        (hfacz, "hFacZ", (levels, ny + 1, nx + 1)),
    ):
        _check_contract_shape(value, name, u, shape)
    for value, name, shape in (
        (dxC_ext, "dxC_ext", (ny + 2, nx + 1)),
        (dxG, "dxG", (ny + 1, nx)),
        (dyG, "dyG", (ny, nx + 1)),
        (dyF, "dyF", (ny, nx)),
        (recip_rA, "recip_rA", (ny, nx)),
        (recip_rAz, "recip_rAz", (ny + 1, nx + 1)),
    ):
        _check_contract_metric(value, name, u, shape)
    for value, name, shape in (
        (maskInC, "maskInC", (ny, nx)),
        (maskInW, "maskInW", (ny, nx + 1)),
        (maskInS, "maskInS", (ny + 1, nx)),
    ):
        if value is not None:
            _check_contract_metric(value, name, u, shape)


def _validate_mr_coeff_contract(values: tuple[torch.Tensor, ...]) -> None:
    (
        hdiv, vort3, strain_bc, vrtDx_x, vrtDy_y, tension_xy, hfacz, l2_d,
        l3_d, l2_z, l3_z, leith2fac, smag2fac, base,
    ) = values
    _contract_check(hdiv.dtype in _SUPPORTED_DTYPES, "inputs must be float32 or float64")
    _contract_check(hdiv.dim() == 4, "hdiv must be four-dimensional")
    if hdiv.dim() != 4:
        return
    batch, levels, ny, nx = hdiv.shape
    _contract_check(nx >= 2, "requires ny >= 2 and nx >= 2")
    _contract_check(ny >= 2, "requires ny >= 2 and nx >= 2")
    _contract_check(batch > 0, "extents must be positive")
    _contract_check(levels > 0, "extents must be positive")
    for value, name, shape in (
        (hdiv, "hdiv", (batch, levels, ny, nx)),
        (vort3, "vort3", (batch, levels, ny + 1, nx + 1)),
        (strain_bc, "strain_bc", (batch, levels, ny + 1, nx + 1)),
        (vrtDx_x, "vrtDx_x", (batch, levels, ny + 1, nx + 2)),
        (vrtDy_y, "vrtDy_y", (batch, levels, ny + 2, nx + 1)),
        (tension_xy, "tension_xy", (batch, levels, ny + 2, nx + 2)),
        (hfacz, "hFacZ", (levels, ny + 1, nx + 1)),
    ):
        _check_contract_shape(value, name, hdiv, shape)
    for value, name, shape in (
        (l2_d, "l2_d", (ny, nx)),
        (l3_d, "l3_d", (ny, nx)),
        (l2_z, "l2_z", (ny + 1, nx + 1)),
        (l3_z, "l3_z", (ny + 1, nx + 1)),
    ):
        _check_contract_metric(value, name, hdiv, shape)
    for value, name in ((leith2fac, "leith2fac"), (smag2fac, "smag2fac"), (base, "base")):
        _check_contract_tensor(value, name, hdiv)
        _contract_check(value.dim() == 0, f"{name} must be a 0-d tensor")


# ---------------------------------------------------------------------------
# Input adapter
# ---------------------------------------------------------------------------


def _scalar(value: float, like: torch.Tensor) -> torch.Tensor:
    """A 0-d ``like``-typed constant, built without a host-to-device copy.

    ``Tensor.new_tensor(python_float)`` materialises the
    scalar on the HOST and copies it in (``cudaMemcpyAsync`` +
    ``cudaStreamSynchronize``); ``torch.full`` passes it as a launch argument to
    ``aten::fill_``.  Same dtype, same round-to-nearest, same bits.
    """
    return torch.full((), value, dtype=like.dtype, device=like.device)


def prepare_stage1_inputs(
    module: Any,
    u: torch.Tensor,
    v: torch.Tensor,
    grid: Any,
) -> tuple[Any, ...]:
    """Build the coefficient kernel's inputs using the eager halo/memo policy.

    The geometry, the curl halos and the memoized geometry constants (hFacZ,
    recip_hFac/masks, length scales) are FETCHED from the very same functions
    ``_divvort_terms`` calls, not recomputed, so a cache hit hands the kernel
    the identical tensors.
    """
    from defgcm.dynamics._horizontal_diffusion_topology import _divvort_curl_halos
    from defgcm.dynamics.viscosity_geometry import (
        _divvort_corner_hfacz,
        _leith_length_scales,
        divvort_geometry,
    )

    dtype, device = u.dtype, u.device
    ny, nx = grid.ny, grid.nx

    # The metric bundle, the scalar-numerator reciprocals (the
    # kernel never divides by a Python scalar), the hFac fields (the
    # ``hFacC is None`` fall-back is declined), the Site 2 memo and maskInC,
    # read by the same function the eager ``_divvort_terms`` calls.
    geom = divvort_geometry(grid, nz=u.shape[-3], dtype=dtype, device=device)
    dyG = geom.dyG
    dxG_ext = geom.dxG
    dxC_row = geom.dxC
    dyC = geom.dyC
    dyF = geom.dyF
    recip_rA = geom.recip_rA
    recip_rAz = geom.recip_rAz
    hFacW = geom.hFacW
    hFacS = geom.hFacS
    recip_hFacC = geom.recip_hFacC
    recip_hFacW = geom.recip_hFacW
    recip_hFacS = geom.recip_hFacS
    maskW = geom.maskW
    maskS = geom.maskS
    maskInC = geom.maskInC

    # The wrapped dxC row block, the east-global-wrap u with its j-halo and
    # the north-global-wrap v scaled by dyC with its i-halo: the eager curl
    # halos, same function, same exchange order.
    dxC_ext, u_ext, v_dyC_ext = _divvort_curl_halos(u, v, grid, dxC=dxC_row, dyC=dyC)

    # :2344-2346 (Site 1 memo).
    hfacz = _divvort_corner_hfacz(grid, hFacW, hFacS, dtype, device)

    # :2914-2925 -- the OBCS interior masks the Leith gradients multiply in.
    maskInW = None
    if grid.masks.maskInW is not None:
        maskInW = grid.masks.maskInW.to(dtype=dtype, device=device)
    maskInS = None
    if grid.masks.maskInS is not None:
        maskInS = grid.masks.maskInS.to(dtype=dtype, device=device)

    # :2411-2467 (Site 3 memo).  ``_leith_length_scales`` reads only these
    # four metric entries plus dtype/device.
    l2_d, l3_d, l2_z, l3_z = _leith_length_scales(
        grid,
        dxF=geom.dxF,
        dyF=dyF,
        dxV=geom.dxV,
        dyU=geom.dyU,
        dtype=dtype,
        device=device,
        ny=ny,
        nx=nx,
    )

    # :2699-2709 -- the two coefficient factors, and :2011 the harmonic base.
    #
    # The 4 explicit waits this file would otherwise own:
    # ``Tensor.new_tensor(python_float)`` materialises the scalar on the HOST
    # and copies it in -- ``cudaMemcpyAsync`` + ``cudaStreamSynchronize``, four
    # per step.  ``torch.full`` passes the value as a launch argument to
    # ``aten::fill_``.  Both round the same Python double to ``dtype`` once with
    # the same round-to-nearest, so every product below keeps its bits.
    ref = l2_d
    pi = _scalar(torch.pi, ref)
    smag2fac = (_scalar(module.visc_c2_smag, ref) / pi) ** 2
    leith2fac = (_scalar(module.visc_c2_leith, ref) / pi) ** 3
    base = _scalar(module.viscosity, u)

    # :1911 -- 0.0 under this client's ``no_slip_sides=False`` domain, passed
    # explicitly rather than assumed so the kernel and the gate share one value.
    side_mask_fac = module.side_drag_factor if module.no_slip_sides else 0.0

    return (
        u,
        v,
        u_ext,
        v_dyC_ext,
        dxC_ext,
        dxG_ext,
        dyG,
        dyF,
        recip_rA,
        recip_rAz,
        hFacW,
        hFacS,
        recip_hFacC,
        maskW,
        maskS,
        hfacz,
        maskInC,
        maskInW,
        maskInS,
        l2_d,
        l3_d,
        l2_z,
        l3_z,
        leith2fac,
        smag2fac,
        base,
        float(side_mask_fac),
        # Carried out for stage 2, not consumed by stage 1.
        recip_hFacW,
        recip_hFacS,
        dxC_row,
        dyC,
    )


#: Number of trailing entries of :func:`prepare_stage1_inputs` that are NOT
#: stage-1 kernel arguments (the float ``side_mask_fac`` is the last argument,
#: the four after it are stage-2 carry-outs).
_STAGE1_CARRY = 4


def stage2_face_flags(grid: Any) -> tuple[bool, bool, bool, bool, bool, bool]:
    """Reproduce ``_divvort_stencil``'s branch selection.

    Single source with the kernel: the same six booleans are handed to the
    launch, so the adapter cannot pick one branch while the kernel runs the
    other.  Under ``config_decline_reason`` there is no distributed neighbour,
    so the ``_has_distributed_neighbor`` disjuncts are all False and the flags
    reduce to ``edge_face_couples``; the calls stay in so the expression keeps
    matching the eager one line for line.
    """
    from defgcm.dynamics._horizontal_diffusion_topology import _has_distributed_neighbor

    x_edge_couples = bool(grid.edge_face_couples("x"))
    y_edge_couples = bool(grid.edge_face_couples("y"))
    west = x_edge_couples or _has_distributed_neighbor(grid, "west")
    east = x_edge_couples or _has_distributed_neighbor(grid, "east")
    south = y_edge_couples or _has_distributed_neighbor(grid, "south")
    north = y_edge_couples or _has_distributed_neighbor(grid, "north")
    return bool(west or east), bool(south or north), west, east, south, north


def prepare_stage2_inputs(
    hDiv_flux: torch.Tensor,
    hz_v3: torch.Tensor,
    grid: Any,
    stage1: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Build the dissipation kernel's inputs, including the halo exchanges.

    The two exchanges below are the ONLY ones between the launches, and they
    run exactly when the eager reference runs them: in the all-faces branches
    of ``HorizontalViscosity._divvort_stencil``, before
    ``_u_divvort_all_from_haloed`` / ``_v_divvort_all_from_haloed``.  On the
    interior branch no halo is built and the kernel reads
    ``hDiv_flux`` directly, matching :1310-1313 / :1332-1334.
    """
    from defgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange

    (
        u_all_faces,
        v_all_faces,
        west,
        east,
        south,
        north,
    ) = stage2_face_flags(grid)

    recip_hFacW, recip_hFacS, dxC_row, dyC = stage1[-_STAGE1_CARRY:]
    dxG_ext = stage1[5]
    dyG = stage1[6]
    maskW = stage1[13]
    maskS = stage1[14]

    hDiv_x = None
    if u_all_faces:
        hDiv_x = exchange(  # :1361-1373
            hDiv_flux,
            grid=grid,
            location="T",
            extent=HaloExtentV1(
                west_left_extent=1,
                east_right_extent=1,
                south_bottom_extent=0,
                north_top_extent=0,
            ),
            fill=FillSpec.ZERO,
        )
    hDiv_y = None
    if v_all_faces:
        hDiv_y = exchange(  # :1394-1406
            hDiv_flux,
            grid=grid,
            location="T",
            extent=HaloExtentV1(
                west_left_extent=0,
                east_right_extent=0,
                south_bottom_extent=1,
                north_top_extent=1,
            ),
            fill=FillSpec.ZERO,
        )

    return (
        hDiv_flux,
        hDiv_x,
        hDiv_y,
        hz_v3,
        dxC_row,
        dyC,
        dxG_ext,
        dyG,
        recip_hFacW,
        recip_hFacS,
        maskW,
        maskS,
        u_all_faces,
        v_all_faces,
        west if u_all_faces else False,
        east if u_all_faces else False,
        south if v_all_faces else False,
        north if v_all_faces else False,
    )


def prepare_multirank_coeff_inputs(
    pre: tuple[torch.Tensor, ...],
    grid: Any,
    stage1: tuple[Any, ...],
) -> tuple[torch.Tensor, ...]:
    """Exchange P outputs exactly where the eager distributed leaves do."""
    from defgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange

    hdiv, vort3, strain_bc, tension, vrtDx, vrtDy = pre
    ny, nx = hdiv.shape[-2:]
    vrtDx_x = exchange(
        vrtDx[..., :, :nx],
        grid=grid,
        location="T",
        extent=HaloExtentV1(1, 1, 0, 0),
        fill=FillSpec.ZERO,
    )
    vrtDy_y = exchange(
        vrtDy[..., :ny, :],
        grid=grid,
        location="T",
        extent=HaloExtentV1(0, 0, 1, 1),
        fill=FillSpec.ZERO,
    )
    tension_xy = exchange(
        tension,
        grid=grid,
        location="T",
        extent=HaloExtentV1.uniform(1),
        fill=FillSpec.ZERO,
    )
    return (
        hdiv,
        vort3,
        strain_bc,
        vrtDx_x,
        vrtDy_y,
        tension_xy,
        stage1[15],
        *stage1[19:26],
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


# nvcc flags this kernel's bitwise equality DEPENDS ON, not preferences:
#   -fmad=false     nvcc defaults to -fmad=true and would contract the
#                   multiply-then-add chains (the flux products in hDiv, the
#                   curl's ``u*dxC`` terms, every ``fac*measure*length``) into
#                   single-rounding FMAs.  The eager reference runs each
#                   multiply and add as its own ATen kernel and cannot
#                   contract, so a contracted kernel silently stops matching.
#   -prec-div=true  every divide in this phase is tensor/tensor,
#                   so all of them must round exactly as ATen's div does.
#   -prec-sqrt=true the two Smagorinsky strain magnitudes take a sqrt.
# Losing any of these changes results with NO error and NO warning.
# These flags must reach the compiler.
REQUIRED_NVCC_FLAGS = ("-O3", "-prec-div=true", "-prec-sqrt=true", "-fmad=false")


def load_extension() -> Any:
    """Build/load the CUDA extension through PyTorch's default cache root."""
    global _EXTENSION, _LOAD_FAILED
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_FAILED:
        raise RuntimeError("hvisc fused extension previously failed to load")
    from torch.utils.cpp_extension import load

    cxx = _find_conda_cxx()
    if cxx is not None:
        os.environ.setdefault("CXX", cxx)
        os.environ.setdefault("CUDAHOSTCXX", cxx)
    flags = list(REQUIRED_NVCC_FLAGS)
    if cxx is not None:
        flags.append(f"-ccbin={cxx}")
    try:
        _EXTENSION = load(
            name="defgcm_hvisc_fused",
            sources=[
                str(_HERE / "extension.cpp"),
                str(_HERE / "kernel.cu"),
                str(_HERE / "kernel_multirank.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=flags,
            with_cuda=True,
            verbose=os.environ.get("DEFGCM_HVISC_FUSED_VERBOSE", "0") == "1",
        )
    except Exception:
        _LOAD_FAILED = True
        raise
    return _EXTENSION


# ---------------------------------------------------------------------------
# Custom ops
# ---------------------------------------------------------------------------


@torch.library.custom_op("defgcm::hvisc_fused_stage1", mutates_args=())
def hvisc_fused_stage1(
    u: torch.Tensor,
    v: torch.Tensor,
    u_ext: torch.Tensor,
    v_dyC_ext: torch.Tensor,
    dxC_ext: torch.Tensor,
    dxG_ext: torch.Tensor,
    dyG: torch.Tensor,
    dyF: torch.Tensor,
    recip_rA: torch.Tensor,
    recip_rAz: torch.Tensor,
    hFacW: torch.Tensor,
    hFacS: torch.Tensor,
    recip_hFacC: torch.Tensor,
    maskW: torch.Tensor,
    maskS: torch.Tensor,
    hfacz: torch.Tensor,
    maskInC: Optional[torch.Tensor],
    maskInW: Optional[torch.Tensor],
    maskInS: Optional[torch.Tensor],
    l2_d: torch.Tensor,
    l3_d: torch.Tensor,
    l2_z: torch.Tensor,
    l3_z: torch.Tensor,
    leith2fac: torch.Tensor,
    smag2fac: torch.Tensor,
    base: torch.Tensor,
    side_mask_fac: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute the coefficient launch; no fallback and no autograd."""
    global _NATIVE_LAUNCHES
    values = (
        u, v, u_ext, v_dyC_ext, dxC_ext, dxG_ext, dyG, dyF, recip_rA, recip_rAz,
        hFacW, hFacS, recip_hFacC, maskW, maskS, hfacz, maskInC, maskInW,
        maskInS, l2_d, l3_d, l2_z, l3_z, leith2fac, smag2fac, base,
    )
    reason = direct_decline_reason(values)
    if reason is not None:
        raise RuntimeError(f"hvisc fused stage 1 is inapplicable: {reason}")
    result = load_extension().hvisc_fused_stage1(
        u, v, u_ext, v_dyC_ext, dxC_ext, dxG_ext, dyG, dyF, recip_rA, recip_rAz,
        hFacW, hFacS, recip_hFacC, maskW, maskS, hfacz, maskInC, maskInW,
        maskInS, l2_d, l3_d, l2_z, l3_z, leith2fac, smag2fac, base,
        float(side_mask_fac),
    )
    _NATIVE_LAUNCHES += 1
    return result[0], result[1], result[2]


@hvisc_fused_stage1.register_fake
def _hvisc_fused_stage1_fake(
    u: torch.Tensor,
    v: torch.Tensor,
    u_ext: torch.Tensor,
    v_dyC_ext: torch.Tensor,
    dxC_ext: torch.Tensor,
    dxG_ext: torch.Tensor,
    dyG: torch.Tensor,
    dyF: torch.Tensor,
    recip_rA: torch.Tensor,
    recip_rAz: torch.Tensor,
    hFacW: torch.Tensor,
    hFacS: torch.Tensor,
    recip_hFacC: torch.Tensor,
    maskW: torch.Tensor,
    maskS: torch.Tensor,
    hfacz: torch.Tensor,
    maskInC: Optional[torch.Tensor],
    maskInW: Optional[torch.Tensor],
    maskInS: Optional[torch.Tensor],
    l2_d: torch.Tensor,
    l3_d: torch.Tensor,
    l2_z: torch.Tensor,
    l3_z: torch.Tensor,
    leith2fac: torch.Tensor,
    smag2fac: torch.Tensor,
    base: torch.Tensor,
    side_mask_fac: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        v, u_ext, v_dyC_ext, dxC_ext, dxG_ext, dyG, dyF, recip_rA, recip_rAz,
        hFacW, hFacS, recip_hFacC, maskW, maskS, hfacz, maskInC, maskInW,
        maskInS, l2_d, l3_d, l2_z, l3_z, leith2fac, smag2fac, base,
        side_mask_fac,
    )
    batch, levels, ny, nx_plus_one = u.shape
    nx = nx_plus_one - 1
    return (
        u.new_empty((batch, levels, ny, nx)),
        u.new_empty((batch, levels, ny, nx)),
        u.new_empty((batch, levels, ny + 1, nx + 1)),
    )


@torch.library.custom_op("defgcm::hvisc_fused_stage2", mutates_args=())
def hvisc_fused_stage2(
    hDiv_flux: torch.Tensor,
    hDiv_x: Optional[torch.Tensor],
    hDiv_y: Optional[torch.Tensor],
    hz_v3: torch.Tensor,
    dxC_row: torch.Tensor,
    dyC: torch.Tensor,
    dxG_ext: torch.Tensor,
    dyG: torch.Tensor,
    recip_hFacW: torch.Tensor,
    recip_hFacS: torch.Tensor,
    maskW: torch.Tensor,
    maskS: torch.Tensor,
    u_all_faces: bool,
    v_all_faces: bool,
    keep_west: bool,
    keep_east: bool,
    keep_south: bool,
    keep_north: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the dissipation launch; no fallback and no autograd."""
    global _NATIVE_LAUNCHES
    values = (
        hDiv_flux, hDiv_x, hDiv_y, hz_v3, dxC_row, dyC, dxG_ext, dyG,
        recip_hFacW, recip_hFacS, maskW, maskS,
    )
    reason = direct_decline_reason(values)
    if reason is not None:
        raise RuntimeError(f"hvisc fused stage 2 is inapplicable: {reason}")
    result = load_extension().hvisc_fused_stage2(
        hDiv_flux, hDiv_x, hDiv_y, hz_v3, dxC_row, dyC, dxG_ext, dyG,
        recip_hFacW, recip_hFacS, maskW, maskS, bool(u_all_faces),
        bool(v_all_faces), bool(keep_west), bool(keep_east), bool(keep_south),
        bool(keep_north),
    )
    _NATIVE_LAUNCHES += 1
    return result[0], result[1]


@hvisc_fused_stage2.register_fake
def _hvisc_fused_stage2_fake(
    hDiv_flux: torch.Tensor,
    hDiv_x: Optional[torch.Tensor],
    hDiv_y: Optional[torch.Tensor],
    hz_v3: torch.Tensor,
    dxC_row: torch.Tensor,
    dyC: torch.Tensor,
    dxG_ext: torch.Tensor,
    dyG: torch.Tensor,
    recip_hFacW: torch.Tensor,
    recip_hFacS: torch.Tensor,
    maskW: torch.Tensor,
    maskS: torch.Tensor,
    u_all_faces: bool,
    v_all_faces: bool,
    keep_west: bool,
    keep_east: bool,
    keep_south: bool,
    keep_north: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        hDiv_x, hDiv_y, hz_v3, dxC_row, dyC, dxG_ext, dyG, recip_hFacW,
        recip_hFacS, maskW, maskS, u_all_faces, v_all_faces, keep_west,
        keep_east, keep_south, keep_north,
    )
    batch, levels, ny, nx = hDiv_flux.shape
    return (
        hDiv_flux.new_empty((batch, levels, ny, nx + 1)),
        hDiv_flux.new_empty((batch, levels, ny + 1, nx)),
    )


@torch.library.custom_op("defgcm::hvisc_fused_mr_pre", mutates_args=())
def hvisc_fused_mr_pre(
    u: torch.Tensor,
    v: torch.Tensor,
    u_ext: torch.Tensor,
    v_dyC_ext: torch.Tensor,
    dxC_ext: torch.Tensor,
    dxG: torch.Tensor,
    dyG: torch.Tensor,
    dyF: torch.Tensor,
    recip_rA: torch.Tensor,
    recip_rAz: torch.Tensor,
    hFacW: torch.Tensor,
    hFacS: torch.Tensor,
    recip_hFacC: torch.Tensor,
    maskW: torch.Tensor,
    maskS: torch.Tensor,
    hfacz: torch.Tensor,
    maskInC: Optional[torch.Tensor],
    maskInW: Optional[torch.Tensor],
    maskInS: Optional[torch.Tensor],
    side_mask_fac: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialise the distributed eager intermediates before communication."""
    global _NATIVE_LAUNCHES
    values = (
        u, v, u_ext, v_dyC_ext, dxC_ext, dxG, dyG, dyF, recip_rA, recip_rAz,
        hFacW, hFacS, recip_hFacC, maskW, maskS, hfacz, maskInC, maskInW, maskInS,
    )
    _validate_mr_pre_contract(values)
    reason = direct_decline_reason(values)
    if reason is not None:
        raise RuntimeError(f"hvisc fused multi-rank P is inapplicable: {reason}")
    result = load_extension().hvisc_fused_mr_pre(*values, float(side_mask_fac))
    _NATIVE_LAUNCHES += 1
    return tuple(result)


@hvisc_fused_mr_pre.register_fake
def _hvisc_fused_mr_pre_fake(
    u, v, u_ext, v_dyC_ext, dxC_ext, dxG, dyG, dyF, recip_rA, recip_rAz,
    hFacW, hFacS, recip_hFacC, maskW, maskS, hfacz, maskInC, maskInW,
    maskInS, side_mask_fac,
):
    values = (
        u, v, u_ext, v_dyC_ext, dxC_ext, dxG, dyG, dyF, recip_rA, recip_rAz,
        hFacW, hFacS, recip_hFacC, maskW, maskS, hfacz, maskInC, maskInW, maskInS,
    )
    _validate_mr_pre_contract(values)
    del side_mask_fac
    batch, levels, ny, nx1 = u.shape
    corner = (batch, levels, ny + 1, nx1)
    center = (batch, levels, ny, nx1 - 1)
    return (
        u.new_empty(center), u.new_empty(corner), u.new_empty(corner),
        u.new_empty(center), u.new_empty(corner), u.new_empty(corner),
    )


@torch.library.custom_op("defgcm::hvisc_fused_mr_coeff", mutates_args=())
def hvisc_fused_mr_coeff(
    hdiv: torch.Tensor,
    vort3: torch.Tensor,
    strain_bc: torch.Tensor,
    vrtDx_x: torch.Tensor,
    vrtDy_y: torch.Tensor,
    tension_xy: torch.Tensor,
    hfacz: torch.Tensor,
    l2_d: torch.Tensor,
    l3_d: torch.Tensor,
    l2_z: torch.Tensor,
    l3_z: torch.Tensor,
    leith2fac: torch.Tensor,
    smag2fac: torch.Tensor,
    base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble coefficients after adapter-side distributed exchanges."""
    global _NATIVE_LAUNCHES
    values = (
        hdiv, vort3, strain_bc, vrtDx_x, vrtDy_y, tension_xy, hfacz, l2_d,
        l3_d, l2_z, l3_z, leith2fac, smag2fac, base,
    )
    _validate_mr_coeff_contract(values)
    reason = direct_decline_reason(values)
    if reason is not None:
        raise RuntimeError(f"hvisc fused multi-rank Q is inapplicable: {reason}")
    result = load_extension().hvisc_fused_mr_coeff(*values)
    _NATIVE_LAUNCHES += 1
    return result[0], result[1], result[2]


@hvisc_fused_mr_coeff.register_fake
def _hvisc_fused_mr_coeff_fake(
    hdiv, vort3, strain_bc, vrtDx_x, vrtDy_y, tension_xy, hfacz, l2_d,
    l3_d, l2_z, l3_z, leith2fac, smag2fac, base,
):
    values = (
        hdiv, vort3, strain_bc, vrtDx_x, vrtDy_y, tension_xy, hfacz, l2_d,
        l3_d, l2_z, l3_z, leith2fac, smag2fac, base,
    )
    _validate_mr_coeff_contract(values)
    batch, levels, ny, nx = hdiv.shape
    return (
        hdiv.new_empty((batch, levels, ny, nx)),
        hdiv.new_empty((batch, levels, ny, nx)),
        hdiv.new_empty((batch, levels, ny + 1, nx + 1)),
    )


# ---------------------------------------------------------------------------
# Counting dispatch shell
# ---------------------------------------------------------------------------


def _record(outcome: str, *, reason: str | None = None) -> None:
    record_dispatch(
        _KEY,
        outcome=outcome,
        variant="leith_smag",
        site=_SITE,
        backend=_BACKEND,
        rank=current_rank(),
        reason=reason,
    )


def hvisc_fused_dispatch(
    module: Any,
    u: torch.Tensor,
    v: torch.Tensor,
    grid: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return ``(u_result, v_result, viscAh_D)``, or ``None`` to run eager.

    ``None`` is the contract with the consumer boundary: the caller keeps the
    frozen eager block, which is therefore the only fallback implementation.
    ``viscAh_D`` comes back because the consumer still owns the ``viscAh_W`` /
    ``viscAh3d_00`` diagnostics (``HorizontalViscosity.evaluate``).
    """
    if not gate(_KEY).effective_value:
        _count("disabled")
        return None

    reason = config_decline_reason(module, grid, u, v)
    if reason is not None:
        _count(f"fallback:{reason}")
        _record("inapplicable", reason=reason)
        return None

    with dispatch_scope(_KEY):
        stage1 = prepare_stage1_inputs(module, u, v, grid)
        stage1_args = stage1[: len(stage1) - _STAGE1_CARRY]
        reason = direct_decline_reason(stage1_args[:-1])
        if reason is not None:
            _count(f"fallback:{reason}")
            _record("inapplicable", reason=reason)
            return None
        try:
            from defgcm.dynamics._horizontal_diffusion_topology import _any_edge_couples

            # Same predicate as the eager _leith_grad_measures branch.  The
            # P/Q kernels read exchanged halos, so a periodic edge of one tile
            # wraps like a rank seam; the single-launch stage-1 kernel zero-pads
            # every outer neighbour and serves only all-wall/open grids.
            if _any_edge_couples(grid):
                pre_args = (*stage1[:19], stage1[26])
                pre = hvisc_fused_mr_pre(*pre_args)
                coeff_args = prepare_multirank_coeff_inputs(pre, grid, stage1)
                viscAh_D, hDiv_flux, hz_v3 = hvisc_fused_mr_coeff(*coeff_args)
            else:
                viscAh_D, hDiv_flux, hz_v3 = hvisc_fused_stage1(*stage1_args)
            stage2 = prepare_stage2_inputs(hDiv_flux, hz_v3, grid, stage1)
            reason = direct_decline_reason(stage2[:12])
            if reason is not None:
                _count(f"fallback:{reason}")
                _record("inapplicable", reason=reason)
                return None
            u_result, v_result = hvisc_fused_stage2(*stage2)
        except Exception as error:
            _count("error")
            _record("error", reason=type(error).__name__)
            raise
    _count("requested-hit")
    _record("hit")
    return u_result, v_result, viscAh_D


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
    "REQUIRED_NVCC_FLAGS",
    "config_decline_reason",
    "direct_decline_reason",
    "dispatch_counts",
    "hvisc_fused_dispatch",
    "hvisc_fused_mr_coeff",
    "hvisc_fused_mr_pre",
    "hvisc_fused_stage1",
    "hvisc_fused_stage2",
    "load_extension",
    "native_launch_count",
    "prepare_stage1_inputs",
    "prepare_stage2_inputs",
    "prepare_multirank_coeff_inputs",
    "reset_dispatch_counts",
    "reset_native_launch_count",
    "stage2_face_flags",
]
