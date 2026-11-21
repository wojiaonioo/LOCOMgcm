"""Bottom drag for momentum equations.

Implements MITgcm's ``mom_u_botdrag_coeff.F`` / ``mom_v_botdrag_coeff.F``
algorithm: linear + quadratic bottom drag applied ONLY to the deepest
wet cell in each water column, with proper ``hFac`` and ``drF``
normalization.

Reference
---------
MITgcm ``mom_u_botdrag_coeff.F`` L101-215, ``mom_fluxform.F`` L684:
    guDiss -= cDrag * uFld * recip_hFacW * recip_drF
where
    cDrag = bottomDragLinear  +  bottomDragQuadratic * sqrt(speed²)
and speed² is computed from the cell-center kinetic energy averaged
to the velocity point (``selectBotDragQuadr=0`` default).

The KE buffer is the one the momentum core hands over: ``MOM_*_BOTDRAG_COEFF``
is called with ``inp_KE=.TRUE.``, skipping the hFac-weighted recompute at
``mom_u_botdrag_coeff.F:137-142``.  The vector-invariant core passes its
``selectKEscheme`` KE (``mom_vecinv.F:284,488``, default 0, unweighted); the
flux-form core always passes the hFac-weighted KEscheme 2
(``mom_fluxform.F:329,675``).  ``BottomDrag.ke_scheme`` carries that choice and
``bind_momentum_core_ke_scheme`` sets it from the composition.
"""
from __future__ import annotations

import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, NamedTuple

import torch

from locomgcm.accel.constant_cache import ConstantCache
from locomgcm.deferred_checks import strict_checks
from locomgcm.geometry import StructuredCGrid
from locomgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange
from locomgcm.geometry.grid import runtime_memo_constant
from locomgcm.state import DiagnosticState, PrognosticState, StateTendency

from .base import MomentumTermClass, TendencyModule, TendencyStepDomain
from .bottom_drag_apply import (
    assemble_bottom_tendency,
    explicit_bottom_drag,
    implicit_bottom_drag,
    split_no_slip_increment,
)
from .bottom_drag_coefficients import (
    bottom_level_coefficient,
    kinetic_energy_center,
    linear_bottom_drag_levels,
    no_slip_bottom_drag_levels,
    no_slip_floor_coefficient,
    non_bottom_zero_sign,
    quadratic_bottom_drag,
    u_bottom_speed,
    v_bottom_speed,
)
from .bottom_geometry import (
    bottom_column_levels,
    bottom_interface_levels,
    bottom_level_mask_from_hfac,
    bottom_recip_dr_c,
    bottom_recip_hfac,
    bottom_thickness,
)
from .vector_invariant_momentum import (
    KE_SCHEMES,
    VectorInvariantMomentumAdvection,
    _compute_ke_center,
    _scheme_name,
)
from .viscosity_geometry import _hfac_c
from locomgcm.dynamics._registry import register_dynamics_module


#: MITgcm KEscheme the flux-form core hands to MOM_{U,V}_BOTDRAG_COEFF
#: (``mom_fluxform.F:329`` ``CALL MOM_CALC_KE(bi,bj,k,2,...)``, used at :675).
_FLUX_FORM_KE_SCHEME = 2


def _ke_halo_fill(grid: StructuredCGrid, axis: Literal["x", "y"]) -> FillSpec:
    """Ghost fill of the KE the domain-edge U/V faces average over.

    MITgcm forms KE on the halo ring from the EXCH'd velocities
    (``mom_calc_ke.F`` loops ``1-OLx..sNx+OLx-1``), and the edge face averages
    ``KE(i)+KE(i-1)`` (``mom_u_botdrag_coeff.F:147-160``,
    ``mom_v_botdrag_coeff.F:147-157``).  An edge face that couples two cells (a
    periodic seam) therefore reads the wrap neighbour; rank seams read the
    neighbour rank through the exchange's distributed transport; an edge that
    couples nothing keeps the replicate fill.
    """
    return FillSpec.WRAP if grid.edge_face_couples(axis) else FillSpec.REPLICATE


def _face_hfac(
    grid: StructuredCGrid, side: str, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """``hFacW`` (``side="w"``) or ``hFacS`` (``side="s"``), all-wet if absent.

    The returned tensor preserves any leading dimensions in the grid's hFac.
    Cube grids use face-first hFac tensors ``[6, nz, y, x]``; treating that
    first axis as a batch dimension and slicing it away would silently apply
    face 0's bottom topology to every face.
    """
    if side == "w":
        hfac = grid.masks.hFacW
        if hfac is not None:
            return hfac.to(dtype=dtype, device=device)
        # Uniform grid: all cells wet, bottom is k = nz-1.
        # hFacW on the u-face has shape [nz, ny, nx+1].
        nx_u = grid.nx + 1 if grid.boundary_topology.open_x else grid.nx
        return torch.ones(grid.nz, grid.ny, nx_u, dtype=dtype, device=device)
    hfac = grid.masks.hFacS
    if hfac is not None:
        return hfac.to(dtype=dtype, device=device)
    ny_v = grid.ny + 1 if grid.boundary_topology.open_y else grid.ny
    return torch.ones(grid.nz, ny_v, grid.nx, dtype=dtype, device=device)


def _assert_single_bottom_interface(mask: torch.Tensor, side: str) -> None:
    """Refuse a column whose wet cells are not contiguous.

    MITgcm applies bottom drag at EVERY interface where
    ``maskW(k) * (1 - maskW(k+1))`` is 1 (``mom_u_botdrag_coeff.F:216-229``,
    last level ``maskW(Nr)``), so a wet-dry-wet column is dragged twice.  The
    two-dimensional form here reads one bottom level per column, which is the
    same physics only while that selector fires at most once.  Rather than
    silently dropping the shallower interface, say so.

    Grids built by ``geometry/masks.py`` cannot produce such a column --
    ``hFacC`` is monotone in ``k`` (:433, the ``hFacMin`` cut at :447-453 only
    trims the deepest cell) and ``hFacW``/``hFacS`` take the elementwise minimum
    of two monotone columns (:502-520) -- so a hit means the hFac producer is
    wrong, not that a configuration needs supporting.
    """
    interfaces = mask.sum(dim=-3)
    if not bool((interfaces > 1).any()):
        return
    offending = int((interfaces > 1).sum())
    worst = int(interfaces.max())
    raise ValueError(
        f"bottom_drag: {offending} hFac{side.upper()} column(s) have up to "
        f"{worst} bottom interfaces, i.e. their wet cells are not contiguous. "
        "MITgcm (mom_u_botdrag_coeff.F:216-229) drags at every "
        "maskW(k)*(1-maskW(k+1)) interface, so such a column takes the drag "
        "more than once; this module reads one bottom level per column and "
        "would drop the shallower interface(s) silently. LOCOMgcm's own "
        "geometry (geometry/masks.py:433,502-520) builds monotone hFac columns "
        "and never produces this, so the hFac fed in here is the thing to fix."
    )


def _bottom_column_constants(
    grid: StructuredCGrid, side: str, *, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Memoised per-column bottom geometry: ``(mask, k_bottom, has_bottom, recip_drF)``.

    ``mask`` is the classic 3-D deepest-wet-cell selector.  ``k_bottom`` is its
    level index per column with a leading singleton level axis (``[..., 1, y,
    x]``), ``has_bottom`` is ``mask`` read at ``k_bottom`` (1 for a column with
    any wet cell, 0 for a fully dry one, which reports ``k_bottom = 0``), and
    ``recip_drF`` is ``1/drF`` read at ``k_bottom``.

    All four are grid constants, so they live in the grid's own runtime cache
    (keyed by side/dtype/device, never by ``id(grid)``) and their lifetime is
    the grid instance's: a grid rebuilt through ``dataclasses.replace`` gets a
    fresh cache dictionary along with its fresh masks.

    All four entries depend on hFac only through the wet/dry pattern and on
    ``layer_thickness``, so they sit on the cacheable side of the
    ``nonlinFreeSurf`` in-place rewrite: ``MaskSet.hfac_time_varying`` states
    that ``UPDATE_SURF_DR`` (LOCOMgcm's ``implicit_free_surface._apply_surf_dr``)
    preserves wet/dry because the ``hFacInf`` floor keeps every wet surface cell
    wet, which is why the bool masks are cached unconditionally too.  hFac
    *magnitudes* are on the other side of that line and are deliberately NOT
    memoised here -- see ``_bottom_recip_hfac``.

    PREMISE, asserted here once per grid: every column's wet cells are
    contiguous, i.e. the MIT per-interface selector fires at most once per
    column.  ``k_bottom`` is "the deepest wet level", which is the same thing
    only under that premise; a wet-dry-wet column has TWO bottom interfaces and
    MITgcm drags at both.  The assertion is a single reduction at cold-build
    time, so it costs nothing per step.
    """
    key = ("bottom_drag.column", side, dtype, str(torch.device(device)))

    def build() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hfac = _face_hfac(grid, side, dtype=dtype, device=device)
        mask = bottom_level_mask_from_hfac(hfac)
        _assert_single_bottom_interface(mask, side)
        k_bottom, has_bottom, recip_dr_f_bottom = bottom_column_levels(
            hfac,
            mask,
            layer_thickness=grid.layer_thickness.to(dtype=dtype, device=device),
        )
        return mask, k_bottom, has_bottom, recip_dr_f_bottom

    return runtime_memo_constant(grid, key, build)  # type: ignore[return-value]


def _bottom_split_viscosity_constants(
    grid: StructuredCGrid, side: str, *, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Memoised ``(bottom_is_internal, k_interface, recip_drC_below[k_bottom])``.

    The split no-slip increment reads the interface viscosity one level BELOW
    the bottom cell, so a bottom at ``k < nz-1`` consumes element ``k+1`` while a
    bottom sitting on the global floor consumes nothing and keeps the ``+0.0``
    the whole-field ``zeros_like`` accumulator left there.  ``k_interface`` is
    clamped so the masked-out floor columns still index in range; the
    ``torch.where`` discards the value they read.
    """
    key = ("bottom_drag.split_viscosity", side, dtype, str(torch.device(device)))

    def build() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, k_bottom, _, _ = _bottom_column_constants(
            grid, side, dtype=dtype, device=device
        )
        nz = int(grid.nz)
        bottom_is_internal, k_interface = bottom_interface_levels(k_bottom, nz=nz)
        if nz > 1:
            recip_dr_c_bottom = bottom_recip_dr_c(
                k_bottom,
                interface_spacing=grid.vertical_coordinate.interface_spacing.to(dtype=dtype, device=device),
                nz=nz,
            )
        else:
            recip_dr_c_bottom = torch.zeros_like(k_bottom, dtype=dtype)
        return bottom_is_internal, k_interface, recip_dr_c_bottom

    return runtime_memo_constant(grid, key, build)  # type: ignore[return-value]


def _bottom_recip_hfac(
    grid: StructuredCGrid,
    side: str,
    k_bottom: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | float:
    """``recip_hFac`` of the bottom cell only, built from the gathered hFac.

    Deliberately not memoised: the nonlinear free surface rewrites ``hFacW[0]``
    every step, so a column whose bottom IS level 0 has a time-varying
    reciprocal.
    """
    hfac = grid.masks.hFacW if side == "w" else grid.masks.hFacS
    if hfac is None:
        # Full-cell grid: the whole-field form multiplied by ones(1, nz, 1, 1).
        return 1.0
    return bottom_recip_hfac(hfac.to(dtype=dtype, device=device), k_bottom)


def _bottom_thickness(
    grid: StructuredCGrid,
    side: str,
    k_bottom: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """``hFac * drF`` of the bottom cell of every face column (0 for a dry column).

    The thickness the drag tendency divides by, as an input of a coefficient
    law.  Read from the current hFac like ``_bottom_recip_hfac``, so it follows
    a nonlinear free surface that rewrites level 0.
    """
    return bottom_thickness(
        _face_hfac(grid, side, dtype=dtype, device=device),
        k_bottom,
        layer_thickness=grid.layer_thickness.to(dtype=dtype, device=device),
    )


def _bottom_mask_w(grid: StructuredCGrid, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return a mask that is 1 only at the deepest wet U-cell.

    Bottom cell at column (j, i): ``hFacW(k, j, i) > 0`` and
    (``k == nz-1`` or ``hFacW(k+1, j, i) == 0``).
    """
    return _bottom_column_constants(grid, "w", dtype=dtype, device=device)[0]


def _bottom_mask_s(grid: StructuredCGrid, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return a mask that is 1 only at the deepest wet V-cell."""
    return _bottom_column_constants(grid, "s", dtype=dtype, device=device)[0]


def _level_coefficients(
    grid: StructuredCGrid,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    linear: float | torch.Tensor,
    vertical_viscosity: float,
    no_slip_bottom: bool,
) -> torch.Tensor:
    """Per-level linear + no-slip-bottom ``cDrag`` (MIT ``mom_u_botdrag_coeff.F``).

    The no-slip term ``viscAr * recip_drC_below * 2`` is on when
    ``no_slip_bottom`` holds and ``vertical_viscosity`` is positive.
    """
    layer_thickness = grid.layer_thickness.to(dtype=dtype, device=device)
    c_drag = linear_bottom_drag_levels(layer_thickness, linear=linear)
    if no_slip_bottom and float(vertical_viscosity) > 0.0:
        c_drag = c_drag + no_slip_bottom_drag_levels(
            layer_thickness,
            grid.vertical_coordinate.interface_spacing.to(dtype=dtype, device=device),
            vertical_viscosity=float(vertical_viscosity),
        )
    return c_drag


def _bottom_level_coefficient(
    grid: StructuredCGrid,
    side: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
    linear: float | torch.Tensor,
    vertical_viscosity: float,
    no_slip_bottom: bool,
) -> torch.Tensor:
    """``0.0 + c_drag[k_bottom]`` per column, using memoised geometry only."""
    _, k_bottom, _, _ = _bottom_column_constants(
        grid, side, dtype=dtype, device=device
    )
    c_drag = _level_coefficients(
        grid,
        dtype=dtype,
        device=device,
        linear=linear,
        vertical_viscosity=vertical_viscosity,
        no_slip_bottom=no_slip_bottom,
    )
    return bottom_level_coefficient(c_drag, k_bottom)


def _non_bottom_zero_sign(
    grid: StructuredCGrid,
    *,
    dtype: torch.dtype,
    device: torch.device,
    linear: float | torch.Tensor,
    vertical_viscosity: float,
    no_slip_bottom: bool,
    linear_term_active: bool,
) -> torch.Tensor:
    """Off-bottom signed-zero scale (``non_bottom_zero_sign``) of this coefficient."""
    if linear_term_active:
        level_coefficient = _level_coefficients(
            grid,
            dtype=dtype,
            device=device,
            linear=linear,
            vertical_viscosity=float(vertical_viscosity),
            no_slip_bottom=bool(no_slip_bottom),
        )
    else:
        # cDrag stayed the ``torch.zeros_like(u)`` accumulator.
        level_coefficient = torch.zeros_like(
            grid.layer_thickness.to(dtype=dtype, device=device).reshape(-1)
        )
    return non_bottom_zero_sign(level_coefficient)


def _with_optional_batch(mask: torch.Tensor, state_tensor: torch.Tensor) -> torch.Tensor:
    if mask.ndim == state_tensor.ndim:
        return mask
    if mask.ndim == state_tensor.ndim - 1:
        return mask.unsqueeze(0)
    raise ValueError(
        f"bottom mask shape {tuple(mask.shape)} cannot align with state tensor "
        f"shape {tuple(state_tensor.shape)}"
    )


class _BottomTerms(NamedTuple):
    """One evaluation of the drag, still split into off-bottom fill + bottom value.

    Both entry points share it: ``evaluate`` assembles it straight away, while
    ``evaluate_with_total_vertical_viscosity`` first folds the split no-slip
    increment into ``u_bottom``/``v_bottom``.  Keeping the bottom value separate
    until assembly is what lets the increment stay two-dimensional.
    """

    u_fill: torch.Tensor
    v_fill: torch.Tensor
    u_bottom: torch.Tensor
    v_bottom: torch.Tensor
    k_w: torch.Tensor
    k_s: torch.Tensor
    has_bottom_w: torch.Tensor
    has_bottom_s: torch.Tensor
    recip_dr_f_w: torch.Tensor
    recip_dr_f_s: torch.Tensor
    recip_hfac_w: torch.Tensor | float
    recip_hfac_s: torch.Tensor | float


def _check_coefficient_law(module: "BottomDrag", law: torch.nn.Module | None) -> None:
    """Refuse a law on a module configured without quadratic drag.

    The quadratic coefficient is also read as a switch:
    ``bind_momentum_core_ke_scheme`` pairs only a quadratic module with the KE
    its momentum core hands over, ``prepare_host_constants`` prewarms that KE
    only for one, and ``active`` (the cube step skips an inactive module) is
    False for a module with no linear, quadratic or no-slip term.  A law takes
    the place of the quadratic term, so all three must see a quadratic module.
    """
    if law is None:
        return
    constant = getattr(law, "constant", None)
    if not isinstance(constant, torch.Tensor) or constant.ndim != 0:
        raise TypeError("a bottom-drag coefficient law needs a 0-d tensor attribute 'constant'")
    if not float(module.quadratic) > 0.0:
        raise ValueError(
            f"BottomDrag.coefficient_law needs quadratic > 0, got {module.quadratic!r}: "
            "bind_momentum_core_ke_scheme leaves a module without quadratic drag at "
            "KEscheme 0 where the flux-form core passes KEscheme 2 "
            "(mom_fluxform.F:329,675), and without linear or no-slip drag the cube "
            "step skips the module (active). Configure the quadratic coefficient; "
            "the law replaces its value."
        )


@dataclass(frozen=True)
@register_dynamics_module("bottom_drag")
class BottomDrag(TendencyModule):
    """Bottom-cell linear + quadratic drag.

    Follows MITgcm ``mom_u_botdrag_coeff.F`` with ``selectBotDragQuadr=0``
    (KE-based speed estimate, the MITgcm default).

    The quadratic-drag speed estimate reads the KE buffer the momentum core
    passes to ``MOM_{U,V}_BOTDRAG_COEFF`` with ``inp_KE=.TRUE.``, so the
    hFac-weighted internal recompute at ``mom_u_botdrag_coeff.F:137-142`` is
    skipped.  MIT bot_drag has no KE control of its own: the vector-invariant
    core passes its ``selectKEscheme`` KE (``mom_vecinv.F:284,488``) and the
    flux-form core the hFac-weighted KEscheme 2 (``mom_fluxform.F:329,675``).
    ``ke_scheme`` records which one this module is paired with; KEscheme 0
    uses ``kinetic_energy_center`` (``mom_calc_ke.F:70-86``), any other value
    the vector-invariant module's own ``_compute_ke_center``.

    Parameters
    ----------
    linear : float
        Linear drag coefficient r_b [m/s].  MITgcm ``bottomDragLinear``.
        Typical value: 2e-4 m/s.
    quadratic : float
        Dimensionless quadratic drag coefficient C_d.
        MITgcm ``bottomDragQuadratic``.  Typical value: 1e-3 to 3e-3.
    implicit : bool
        If True, return a backward-Euler equivalent tendency for the local drag
        rate, using ``dt``.  This is a stability option for real-data runs with
        very shallow wet cells; the default remains the explicit MIT-style
        tendency used by alignment tests.
    dt : float, optional
        Time step in seconds required when ``implicit=True``.
    ke_scheme : int
        MITgcm KEscheme (0-3, ``mom_calc_ke.F``) of the KE the momentum core
        hands over.  Model assembly sets it through
        ``bind_momentum_core_ke_scheme``; the default 0 is the
        vector-invariant default and the value a standalone module uses.
    coefficient_law : torch.nn.Module, optional
        Replaces the linear and quadratic coefficients; None keeps
        ``cDrag = linear + no-slip + quadratic * |u|``.  A law has a 0-d tensor
        ``constant``, added where ``linear`` is, and is called as
        ``law(speed, thickness)`` with the KE speed and ``hFac * drF`` of the
        bottom faces; its result is added where ``quadratic * |u|`` is.  The
        no-slip term, the bottom-face selection and the tendency are unchanged.
        Requires ``quadratic > 0`` (``_check_coefficient_law``).  Comparison,
        hashing and repr leave it out, so they describe the configuration.
        ``locomgcm.learned.bottom_drag.LearnedDragInjection`` sets it.
    """

    linear: float | torch.Tensor = 0.0
    quadratic: float = 0.0
    implicit: bool = False
    dt: float | None = None
    vertical_viscosity: float = 0.0
    no_slip_bottom: bool = True
    ke_scheme: int = 0
    name: str = "bottom_drag"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW
    # An init field: ``checkpoint.recompute._bind`` re-derives the init=False
    # fields from a ``replace`` copy, which would reset an init=False law.
    coefficient_law: torch.nn.Module | None = field(default=None, compare=False, repr=False)
    momentum_term_class = MomentumTermClass.BOTTOM_DRAG
    # Topology capability (locomgcm/grid_support.py).
    # Cube claim backed by a bitwise registry-injection check.
    grid_support = ("latlon", "cube")
    # Whether any drag term contributes, decided once here from the configured
    # coefficients (a host read at construction).  The cube step gates the
    # module on this Python bool: reading ``linear`` inside the step is a
    # ``Tensor.item()`` in the traced region and breaks the compiled graph.
    active: bool = field(init=False, default=False)
    _linear_constants: ConstantCache = field(
        default_factory=ConstantCache, init=False, repr=False, compare=False,
    )
    _linear_source_stamp: tuple | None = field(
        default=None, init=False, repr=False, compare=False,
    )
    _linear_cacheable: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.linear, torch.Tensor):
            object.__setattr__(self, "linear", torch.as_tensor(self.linear, dtype=torch.float64))
        if self.linear.ndim != 0 or not bool(torch.isfinite(self.linear)):
            raise ValueError("BottomDrag.linear must be a finite scalar")
        ke_codes = tuple(code for _, code in KE_SCHEMES)
        if isinstance(self.ke_scheme, bool) or self.ke_scheme not in ke_codes:
            raise ValueError(
                "BottomDrag.ke_scheme must be a MITgcm KEscheme in "
                "{" + ", ".join(str(code) for code in ke_codes) + "} "
                f"(mom_calc_ke.F), got {self.ke_scheme!r}"
            )
        object.__setattr__(
            self,
            "active",
            bool(self.linear != 0)
            or float(self.quadratic) != 0.0
            or (bool(self.no_slip_bottom) and float(self.vertical_viscosity) > 0.0),
        )
        _check_coefficient_law(self, self.coefficient_law)
        coefficient = self.linear if self.coefficient_law is None else self.coefficient_law.constant
        object.__setattr__(self, "_linear_cacheable", not coefficient.is_inference())

    def _linear_for(self, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        coefficient = self.linear if self.coefficient_law is None else self.coefficient_law.constant
        if not torch.compiler.is_compiling():
            # Inference tensors have no version counter.  Resolve this on the
            # host: Tensor.is_inference itself is not traceable in all modes.
            object.__setattr__(self, "_linear_cacheable", (
                not coefficient.is_inference()
                and not torch._C._are_functorch_transforms_active()
            ))
        # Active coefficients keep the original conversion in this evaluation's
        # graph; only constants can outlive an evaluation or a replay segment.
        if (not self._linear_cacheable or coefficient.requires_grad
                or torch.autograd.forward_ad.unpack_dual(coefficient).tangent is not None):
            return coefficient
        if not torch.compiler.is_compiling():
            stamp = self._linear_source_stamp
            if stamp is None or stamp[0]() is not coefficient or stamp[1] != coefficient._version:
                self._linear_constants.clear()
                object.__setattr__(self, "_linear_source_stamp", (
                    weakref.ref(coefficient), coefficient._version,
                ))
        return self._linear_constants.get_or_build(
            (dtype, str(torch.device(device))),
            lambda: torch.as_tensor(coefficient, dtype=dtype, device=device),
            variant="bottom_drag_linear",
            site="BottomDrag._linear_for",
            tensor_only=True,
        )

    def to_runtime(
        self, device: torch.device | str, dtype: torch.dtype | None = None,
    ) -> "BottomDrag":
        self._linear_for(dtype=self.linear.dtype if dtype is None else dtype, device=device)
        return self

    def prepare_host_constants(
        self,
        grid: StructuredCGrid,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        """Eager pre-step: build the bottom-column constants of both faces.

        Warm the same memo that _bottom_terms reads, keeping cold construction
        outside the traced region. Weighted KE schemes prepare the hFac and
        area accessors directly without allocating dummy velocity fields.
        """
        if not self.active:
            return
        dev = torch.device(device)
        self._linear_for(dtype=dtype, device=dev)
        for side in ("w", "s"):
            _bottom_column_constants(grid, side, dtype=dtype, device=dev)
        self.require_level_sign_constant(grid, dtype=dtype, device=dev)
        if float(self.quadratic) > 0.0:
            ke_scheme = _scheme_name(KE_SCHEMES, self.ke_scheme, "BottomDrag.ke_scheme")
            if ke_scheme in ("area_weighted", "area_hfac_weighted"):
                grid.metrics.cell_area_for(dtype, dev)
                grid.metrics.rAw_for(dtype, dev)
                grid.metrics.rAs_for(dtype, dev)
            if ke_scheme in ("hfac_weighted", "area_hfac_weighted"):
                grid.hfac_w_for(dtype, dev)
                grid.hfac_s_for(dtype, dev)
                if grid.masks.hFacC is not None:
                    _hfac_c(grid, dtype, dev, site="vector_invariant_momentum._hfac_c")

    def require_level_sign_constant(
        self,
        grid: StructuredCGrid,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        """Quadratic drag needs every level's linear and no-slip coefficient >= 0.

        **Eager.**  With a negative level coefficient the sign of the
        off-bottom zero follows the running speed, which the level-constant
        scale of ``non_bottom_zero_sign`` cannot reproduce, so the step refuses
        such a coefficient.  The coefficient is a control: it changes where it
        is bound (``__post_init__``, ``checkpoint.recompute._bind``,
        :meth:`set_coefficient_law`) or updated in place by an optimizer, never
        inside a step.  The classification is therefore read here, in the
        eager pre-step, once per coefficient object and version; the traced
        step carries no assertion on it.
        """
        law = self.coefficient_law
        if law is None and not float(self.quadratic) > 0.0:
            return
        coefficient = self.linear if law is None else law.constant
        thickness = grid.layer_thickness
        dev = torch.device(device)
        settings = (dtype, dev, float(self.vertical_viscosity), bool(self.no_slip_bottom))
        checked = self.__dict__.get("_level_sign_checked")
        if (checked is not None and checked[0]() is coefficient
                and checked[1] == coefficient._version and checked[2]() is thickness
                and checked[3] == settings):
            return
        base = _level_coefficients(
            grid,
            dtype=dtype,
            device=dev,
            linear=self._linear_for(dtype=dtype, device=dev),
            vertical_viscosity=float(self.vertical_viscosity),
            no_slip_bottom=bool(self.no_slip_bottom),
        )
        if not bool(torch.all(base >= 0)):
            raise ValueError(
                "BottomDrag with quadratic drag requires a non-negative "
                "linear/no-slip level coefficient: with a negative one the "
                "sign of the off-bottom zero depends on the running speed "
                "and cannot be reproduced without a whole-field pass"
            )
        object.__setattr__(self, "_level_sign_checked", (
            weakref.ref(coefficient), coefficient._version, weakref.ref(thickness), settings))

    def set_coefficient_law(self, law: torch.nn.Module | None) -> None:
        """Set ``coefficient_law`` on this instance; ``None`` clears it.

        The model holds this object (``dynamics_modules``, its partitions, the
        cube template in the stepper cache), so the law goes onto it rather than
        onto a ``replace`` copy, the way ``checkpoint.recompute._bind`` binds a
        control; ``frozen=True`` leaves ``object.__setattr__`` as the only
        in-place write.  As instance state the law is visible to Dynamo's
        guards on ``self``.
        """
        _check_coefficient_law(self, law)
        object.__setattr__(self, "coefficient_law", law)

    def _bottom_terms(
        self, state: PrognosticState, grid: StructuredCGrid, *,
        kappa_ru_floor: torch.Tensor | None, kappa_rv_floor: torch.Tensor | None,
    ) -> "_BottomTerms":
        """``kappa_r*_floor`` is the interface Nr+1 viscosity a vertical mixing
        scheme sets (``EvaluationInputs``); None reads ``vertical_viscosity``."""
        u = state.u.data
        v = state.v.data
        dtype = u.dtype
        device = u.device

        # ── Bottom column geometry (grid constants) ──
        # The drag is a two-dimensional bottom-cell quantity, so everything is
        # read at k_bottom and written back only there; the pre- form built
        # 3-D masks and multiplied whole fields.  Operand values and their
        # multiplication order are unchanged, so the bottom cells stay bitwise.
        _, k_w, has_bot_w, recip_dr_f_w = _bottom_column_constants(
            grid, "w", dtype=dtype, device=device
        )
        _, k_s, has_bot_s, recip_dr_f_s = _bottom_column_constants(
            grid, "s", dtype=dtype, device=device
        )
        k_w = _with_optional_batch(k_w, u)
        k_s = _with_optional_batch(k_s, v)
        has_bot_w = _with_optional_batch(has_bot_w, u)
        has_bot_s = _with_optional_batch(has_bot_s, v)
        recip_dr_f_w = _with_optional_batch(recip_dr_f_w, u)
        recip_dr_f_s = _with_optional_batch(recip_dr_f_s, v)

        recip_hfw_bot = _bottom_recip_hfac(grid, "w", k_w, dtype=dtype, device=device)
        recip_hfs_bot = _bottom_recip_hfac(grid, "s", k_s, dtype=dtype, device=device)

        # ── Build drag coefficient cDrag at each bottom cell ──
        # A coefficient law (``coefficient_law``) replaces the configured
        # linear and quadratic coefficients; the no-slip term stays.
        law = self.coefficient_law
        c_lin = self._linear_for(dtype=dtype, device=device)

        c_quad_val = float(self.quadratic) if self.quadratic != 0.0 else None
        uses_speed = law is not None or (c_quad_val is not None and c_quad_val > 0)

        linear_term_active = c_lin is not None or (
            self.no_slip_bottom and float(self.vertical_viscosity) > 0.0
        )
        cDrag_u: torch.Tensor | float = 0.0
        cDrag_v: torch.Tensor | float = 0.0
        if linear_term_active:
            cDrag_u = _with_optional_batch(
                _bottom_level_coefficient(
                    grid,
                    "w",
                    dtype=dtype,
                    device=device,
                    linear=c_lin,
                    vertical_viscosity=float(self.vertical_viscosity),
                    no_slip_bottom=bool(self.no_slip_bottom),
                ),
                u,
            )
            cDrag_v = _with_optional_batch(
                _bottom_level_coefficient(
                    grid,
                    "s",
                    dtype=dtype,
                    device=device,
                    linear=c_lin,
                    vertical_viscosity=float(self.vertical_viscosity),
                    no_slip_bottom=bool(self.no_slip_bottom),
                ),
                v,
            )
        if kappa_ru_floor is not None and self.no_slip_bottom:
            # calc_viscosity.F:121-130: a bottom cell on the last level reads
            # the copy of interface Nr; one above it reads a dry face, whose
            # merged viscosity is the background (mom_u_botdrag_coeff.F:72-75).
            thickness = grid.layer_thickness.to(dtype=dtype, device=device)
            last = grid.nz - 1
            cDrag_u = torch.where(
                k_w == last, no_slip_floor_coefficient(kappa_ru_floor, thickness, linear=c_lin), cDrag_u)
            cDrag_v = torch.where(
                k_s == last, no_slip_floor_coefficient(kappa_rv_floor, thickness, linear=c_lin), cDrag_v)

        # ── Quadratic drag: speed² from KE ──
        if uses_speed:
            ke_scheme = _scheme_name(KE_SCHEMES, self.ke_scheme, "BottomDrag.ke_scheme")
            if ke_scheme == "unweighted":
                ke = kinetic_energy_center(u, v)
            elif ke_scheme in ("area_weighted", "hfac_weighted", "area_hfac_weighted"):
                ke = _compute_ke_center(u, v, grid, select_ke_scheme=self.ke_scheme)
            u_east_extent = u.shape[-1] - ke.shape[-1]
            v_north_extent = v.shape[-2] - ke.shape[-2]
            if u_east_extent not in (0, 1) or v_north_extent not in (0, 1):
                raise ValueError(
                    "bottom_drag KE interpolation expects U/V face dimensions "
                    "to match center dimensions or exceed them by one"
                )
            # KE and its halo stay three-dimensional: the halo column a U point
            # needs is read at THIS rank's k_bottom, which the neighbour cannot
            # gather on our behalf.  The bottom read happens right after the
            # exchange, so the averaging/clamp/sqrt run in 2-D.
            # Average KE to U-points: 0.5*(KE(i-1,j) + KE(i,j))
            # KE has shape [B, nz, ny, nx]
            ke_halo_x = exchange(
                ke,
                grid=grid,
                location="T",
                extent=HaloExtentV1(
                    west_left_extent=1,
                    east_right_extent=u_east_extent,
                    south_bottom_extent=0,
                    north_top_extent=0,
                ),
                fill=_ke_halo_fill(grid, "x"),
            )
            speed_floor = torch.finfo(dtype).tiny
            speed_u = u_bottom_speed(ke_halo_x, k_w, speed_floor=speed_floor)

            # Average KE to V-points: 0.5*(KE(i,j-1) + KE(i,j))
            ke_halo_y = exchange(
                ke,
                grid=grid,
                location="T",
                extent=HaloExtentV1(
                    west_left_extent=0,
                    east_right_extent=0,
                    south_bottom_extent=1,
                    north_top_extent=v_north_extent,
                ),
                fill=_ke_halo_fill(grid, "y"),
            )
            speed_v = v_bottom_speed(ke_halo_y, k_s, speed_floor=speed_floor)

            if law is None:
                cDrag_u = cDrag_u + quadratic_bottom_drag(speed_u, quadratic=c_quad_val)
                cDrag_v = cDrag_v + quadratic_bottom_drag(speed_v, quadratic=c_quad_val)
            else:
                cDrag_u = cDrag_u + law(
                    speed_u,
                    _bottom_thickness(grid, "w", k_w, dtype=dtype, device=device),
                )
                cDrag_v = cDrag_v + law(
                    speed_v,
                    _bottom_thickness(grid, "s", k_s, dtype=dtype, device=device),
                )

        # Apply: du/dt = -cDrag * u * recip_hFacW * recip_drF * bottom_mask.
        # For real-data shallow shelves this explicit drag can violate the
        # local friction time scale.  The opt-in implicit form keeps the same
        # current-state drag coefficient but returns the backward-Euler
        # equivalent tendency: u_new = u / (1 + dt * rate).
        if self.implicit:
            if self.dt is None or float(self.dt) <= 0.0:
                raise ValueError("BottomDrag implicit=True requires dt > 0")
            dt = torch.tensor(float(self.dt), dtype=dtype, device=device)
            u_tendency, u_bottom_tendency = implicit_bottom_drag(
                u,
                k_bottom=k_w,
                c_drag=cDrag_u,
                recip_hfac=recip_hfw_bot,
                recip_dr_f=recip_dr_f_w,
                has_bottom=has_bot_w,
                dt=dt,
            )
            v_tendency, v_bottom_tendency = implicit_bottom_drag(
                v,
                k_bottom=k_s,
                c_drag=cDrag_v,
                recip_hfac=recip_hfs_bot,
                recip_dr_f=recip_dr_f_s,
                has_bottom=has_bot_s,
                dt=dt,
            )
        else:
            zero_scale = _non_bottom_zero_sign(
                grid,
                dtype=dtype,
                device=device,
                linear=c_lin,
                vertical_viscosity=float(self.vertical_viscosity),
                no_slip_bottom=bool(self.no_slip_bottom),
                linear_term_active=linear_term_active,
            )
            if uses_speed and not torch.compiler.is_compiling():
                # A compiled step was preceded by its eager pre-step, which
                # established this (prepare_host_constants); an eager call
                # outside a prepared step establishes it here.
                self.require_level_sign_constant(grid, dtype=dtype, device=device)
            u_tendency, u_bottom_tendency = explicit_bottom_drag(
                u,
                k_bottom=k_w,
                c_drag=cDrag_u,
                recip_hfac=recip_hfw_bot,
                recip_dr_f=recip_dr_f_w,
                has_bottom=has_bot_w,
                zero_scale=zero_scale,
            )
            v_tendency, v_bottom_tendency = explicit_bottom_drag(
                v,
                k_bottom=k_s,
                c_drag=cDrag_v,
                recip_hfac=recip_hfs_bot,
                recip_dr_f=recip_dr_f_s,
                has_bottom=has_bot_s,
                zero_scale=zero_scale,
            )

        return _BottomTerms(
            u_fill=u_tendency,
            v_fill=v_tendency,
            u_bottom=u_bottom_tendency,
            v_bottom=v_bottom_tendency,
            k_w=k_w,
            k_s=k_s,
            has_bottom_w=has_bot_w,
            has_bottom_s=has_bot_s,
            recip_dr_f_w=recip_dr_f_w,
            recip_dr_f_s=recip_dr_f_s,
            recip_hfac_w=recip_hfw_bot,
            recip_hfac_s=recip_hfs_bot,
        )

    @staticmethod
    def _assemble(state: PrognosticState, terms: "_BottomTerms") -> StateTendency:
        """Write the bottom values into the off-bottom fill, exactly once.

        The fill is a fresh tensor nobody else holds, so the scatter is in place;
        the split-viscosity entry point folds its increment into ``u_bottom``
        BEFORE calling this, rather than patching an already-assembled field --
        gathering back out of the assembled tensor and mutating it again is what
        autograd (rightly) rejects.
        """
        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=assemble_bottom_tendency(terms.u_fill, terms.u_bottom, k_bottom=terms.k_w),
            v=assemble_bottom_tendency(terms.v_fill, terms.v_bottom, k_bottom=terms.k_s),
            temp=torch.zeros_like(state.temp.data),
            salt=torch.zeros_like(state.salt.data),
        )

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        inputs = diagnostics.inputs
        return self._assemble(state, self._bottom_terms(
            state, grid,
            kappa_ru_floor=None if inputs is None else inputs.kappa_ru_floor,
            kappa_rv_floor=None if inputs is None else inputs.kappa_rv_floor,
        ))

    def evaluate_with_total_vertical_viscosity(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
        *,
        total_vertical_viscosity_u: torch.Tensor,
        total_vertical_viscosity_v: torch.Tensor,
    ) -> StateTendency:
        """Evaluate drag with split local/global no-slip viscosity ownership.

        ``total_vertical_viscosity_u/v`` are length-``N`` total background+KPP
        interface fields on W/S points.  A local bottom at level ``k<N-1``
        consumes element ``k+1``.  A bottom at the global last level consumes
        only the configured background carried by ``vertical_viscosity``;
        element ``N`` is neither required nor accessed and the last internal
        KPP value is never copied to the global floor.
        """
        for name, value, target in (
            ("total_vertical_viscosity_u", total_vertical_viscosity_u, state.u.data),
            ("total_vertical_viscosity_v", total_vertical_viscosity_v, state.v.data),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"{name} must have shape {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            if value.dtype != target.dtype:
                raise TypeError(f"{name} must have dtype {target.dtype}, got {value.dtype}")
            if value.device != target.device:
                raise ValueError(f"{name} must be on device {target.device}")
            if strict_checks() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain only finite values")

        del diagnostics
        # The vertical mixing schemes are refused with momentum implicit
        # vertical advection (steppers/ifs/columns.validate_vertical_mixing).
        terms = self._bottom_terms(state, grid, kappa_ru_floor=None, kappa_rv_floor=None)
        if not self.no_slip_bottom:
            return self._assemble(state, terms)

        dtype = state.u.data.dtype
        device = state.u.data.device
        bot_w = _with_optional_batch(
            _bottom_mask_w(grid, dtype=dtype, device=device),
            state.u.data,
        )
        bot_s = _with_optional_batch(
            _bottom_mask_s(grid, dtype=dtype, device=device),
            state.v.data,
        )
        background = float(self.vertical_viscosity)
        # Configuration precondition, not a state check: whether the bathymetry
        # puts a bottom on the last level is a property of the grid and cannot
        # change during a run, yet the check ran every step.  **Strict level
        # only** -- two whole-field reductions and a host branch inside the
        # traced step. Nothing downstream
        # reads the verdict; it exists to reject the configuration.
        if strict_checks() and background <= 0.0:
            global_bottom_present = bool(
                torch.any(bot_w[..., -1, :, :] != 0)
                or torch.any(bot_s[..., -1, :, :] != 0)
            )
            if global_bottom_present:
                raise ValueError(
                    "no-slip global bottom requires positive "
                    "BottomDrag.vertical_viscosity background"
                )

        # The increment is a bottom-cell quantity too: read the interface
        # viscosity one level below the bottom, form the coefficient in 2-D and
        # add it into the base tendency at the bottom level only.  Operand
        # values and multiplication order are the whole-field ones.
        #
        # Off the bottom the whole-field form added a signed zero to the base's
        # signed zero; here the base is left alone.  The two agree bit for bit
        # whenever ``total_vertical_viscosity >= vertical_viscosity`` -- the
        # documented background+KPP contract, under which the increment is
        # non-negative -- and they agree unconditionally once the module output
        # reaches a ``StateTendency`` accumulator, which is initialised at
        # ``+0.0`` and therefore normalises either zero.  Reproducing the sign
        # in the other case would need the whole-field ``sign(total - background)``,
        # i.e. a device->host reduction on the step's critical path.
        u_data = state.u.data
        v_data = state.v.data
        internal_w, k_iface_w, recip_dr_c_w = _bottom_split_viscosity_constants(
            grid, "w", dtype=dtype, device=device
        )
        internal_s, k_iface_s, recip_dr_c_s = _bottom_split_viscosity_constants(
            grid, "s", dtype=dtype, device=device
        )
        internal_w = _with_optional_batch(internal_w, u_data)
        internal_s = _with_optional_batch(internal_s, v_data)
        k_iface_w = _with_optional_batch(k_iface_w, u_data)
        k_iface_s = _with_optional_batch(k_iface_s, v_data)
        recip_dr_c_w = _with_optional_batch(recip_dr_c_w, u_data)
        recip_dr_c_s = _with_optional_batch(recip_dr_c_s, v_data)
        # `terms.u_bottom` is still a standalone 2-D tensor at this point, so
        # the increment folds into it before the single assembly scatter.
        u_delta = split_no_slip_increment(
            u_data,
            total_vertical_viscosity_u,
            k_bottom=terms.k_w,
            k_interface=k_iface_w,
            bottom_is_internal=internal_w,
            background=background,
            recip_dr_c=recip_dr_c_w,
            recip_hfac=terms.recip_hfac_w,
            recip_dr_f=terms.recip_dr_f_w,
            has_bottom=terms.has_bottom_w,
        )
        v_delta = split_no_slip_increment(
            v_data,
            total_vertical_viscosity_v,
            k_bottom=terms.k_s,
            k_interface=k_iface_s,
            bottom_is_internal=internal_s,
            background=background,
            recip_dr_c=recip_dr_c_s,
            recip_hfac=terms.recip_hfac_s,
            recip_dr_f=terms.recip_dr_f_s,
            has_bottom=terms.has_bottom_s,
        )
        return self._assemble(
            state,
            terms._replace(
                u_bottom=terms.u_bottom + u_delta,
                v_bottom=terms.v_bottom + v_delta,
            ),
        )


def bind_momentum_core_ke_scheme(
    modules: Sequence[TendencyModule],
) -> tuple[TendencyModule, ...]:
    """Pair every quadratic ``BottomDrag`` with the KE its momentum core computes.

    MITgcm's bottom drag reads the KE buffer the momentum core passes with
    ``inp_KE=.TRUE.``: the vector-invariant core its ``selectKEscheme`` KE
    (``mom_vecinv.F:284,488``), the flux-form core KEscheme 2
    (``mom_fluxform.F:329,675``).  When the composition holds such a core
    (``VectorInvariantMomentumAdvection``, ``FluxFormMomentum`` or
    ``MomentumAdvection``), a quadratic module still at the default 0 is bound
    to the core's scheme; a module configured with a different non-zero scheme,
    or cores that disagree, raise.  Without a core (a bare ``FCoriolis`` is
    used for both forms, so it names neither) the configured value stands.
    """
    from .flux_form_momentum import FluxFormMomentum
    from .momentum_advection import MomentumAdvection

    if not any(
        isinstance(module, BottomDrag) and float(module.quadratic) > 0.0
        for module in modules
    ):
        return tuple(modules)
    schemes: set[int] = set()
    for module in modules:
        if isinstance(module, VectorInvariantMomentumAdvection):
            schemes.add(int(module.select_ke_scheme))
        elif isinstance(module, (FluxFormMomentum, MomentumAdvection)):
            schemes.add(_FLUX_FORM_KE_SCHEME)
    if not schemes:
        return tuple(modules)
    if len(schemes) > 1:
        raise ValueError(
            "bottom drag KE: the momentum modules of this composition hand "
            f"MOM_*_BOTDRAG_COEFF different KE schemes {sorted(schemes)}"
        )
    (scheme,) = schemes
    bound: list[TendencyModule] = []
    for module in modules:
        if (
            isinstance(module, BottomDrag)
            and float(module.quadratic) > 0.0
            and module.ke_scheme != scheme
        ):
            if module.ke_scheme != 0:
                raise ValueError(
                    f"BottomDrag.ke_scheme={module.ke_scheme} contradicts the "
                    f"momentum core, which hands over KEscheme {scheme} "
                    "(mom_vecinv.F:284 / mom_fluxform.F:329)"
                )
            module = replace(module, ke_scheme=scheme)
        bound.append(module)
    return tuple(bound)
