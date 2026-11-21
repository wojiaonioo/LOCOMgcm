from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import torch

from locomgcm.geometry import StructuredCGrid
from locomgcm.geometry.grid import _compute_interior_mask
from locomgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange
from locomgcm.operators.c_grid import tracer_flux_divergence_to_center
from locomgcm.operators.reconstructions._boundary import prepare_obc_tracer

from .piecewise_polynomial import (
    StencilCellInputs,
    ppm_transport_flux,
    pqm_transport_flux,
    vertical_ppm_flux,
    vertical_pqm_flux,
)
from .tracer_reconstruction import (
    centered4_transport_flux,
    centered_face_flux,
    dst2_face_flux,
    dst3_transport_flux,
    dst3fl_transport_flux,
    face_courant_number,
    first_order_upwind_flux,
    fluxlimit_transport_flux,
    os7mp_transport_flux,
    upwind3_transport_flux,
    upwind_face_flux,
    vertical_advective_flux_dst3,
    vertical_advective_flux_dst3fl,
    vertical_centered4_flux,
    vertical_centered_flux,
    vertical_dst2_flux,
    vertical_fluxlimit_flux,
    vertical_os7mp_flux,
    vertical_upwind3_flux,
    vertical_upwind_flux,
)


@dataclass(frozen=True)
class TracerAdvectionScheme:
    """One tracer advection scheme, as configuration, translators and adapters read it.

    ``name`` is the configuration value; ``mit_code`` the MITgcm ``GAD.h`` code,
    shared by ``tempAdvScheme`` and ``tempVertAdvScheme``; ``halo`` the cells a
    face stencil reads on each side of the face (MITgcm ``GAD_advScheme_OlMin``,
    ``gad_init_fixed.F:44-60``); ``needs_dt`` whether the face flux reads the
    Courant number ``|u dt / dx|``, so the tracer time step must be positive;
    ``adams_bashforth`` whether MITgcm steps a tracer advected by this scheme
    with Adams-Bashforth and forces its multi-dimensional advection off
    (``AdamsBashforthGt``, ``tempMultiDimAdvec``, ``gad_init_fixed.F:126-163``);
    ``reads_physical_face_mask`` whether the horizontal face flux also reads
    the physical face masks ``maskW(i-1)``, ``maskW(i+1)``, which differ from
    ``maskLocW = maskW*maskInW`` on an open-boundary strip, so an adapter
    fetches that window only for such a scheme; ``reads_cell_geometry``
    whether the face flux reads the cell masks ``maskC``, the cell widths and
    the face spacings of its stencil and the signed face velocity
    (:class:`~locomgcm.operators.fluxes.piecewise_polynomial.StencilCellInputs`),
    which an adapter fetches only for such a scheme; ``single_rhs`` whether
    MITgcm's single-RHS path (``GAD_CALC_RHS``) has the scheme, so it can run
    without the multi-dimensional sweep (PPM and PQM stop there,
    ``gad_calc_rhs.F:295-296``, ``:424-425``).  For PPM with the weno limiter
    ``halo`` is the stencil reach, one cell more than MITgcm's overlap minimum
    (its oscillation indicators read one cell beyond the edge values).  The
    weno schemes (42, 52) take the centred oscillation indicator on every
    cell, which is MITgcm's result when the overlap is at least 4; with
    ``OLx`` or ``OLy`` 3 MITgcm uses the one-sided indicator on the outermost
    overlap cell (``gad_osc_hat_x.F:50-68``), so its result depends on the tile
    layout, and such a build is out of scope (the translator refuses it).
    """

    name: str
    mit_code: int
    halo: int
    needs_dt: bool
    adams_bashforth: bool
    reads_physical_face_mask: bool
    reads_cell_geometry: bool
    single_rhs: bool


#: Every tracer advection scheme the model names.  The name-to-code
#: correspondence is written only here.
TRACER_ADVECTION_SCHEMES = (
    # ENUM_UPWIND_1RST
    TracerAdvectionScheme(
        "upwind", 1, halo=1, needs_dt=False, adams_bashforth=False,
        reads_physical_face_mask=False, reads_cell_geometry=False,
        single_rhs=True,
    ),
    # ENUM_CENTERED_2ND
    TracerAdvectionScheme(
        "centered", 2, halo=1, needs_dt=False, adams_bashforth=True,
        reads_physical_face_mask=False, reads_cell_geometry=False,
        single_rhs=True,
    ),
    # ENUM_UPWIND_3RD
    TracerAdvectionScheme(
        "upwind3", 3, halo=2, needs_dt=False, adams_bashforth=True,
        reads_physical_face_mask=False, reads_cell_geometry=False,
        single_rhs=True,
    ),
    # ENUM_CENTERED_4TH
    TracerAdvectionScheme(
        "centered4", 4, halo=2, needs_dt=False, adams_bashforth=True,
        reads_physical_face_mask=True, reads_cell_geometry=False,
        single_rhs=True,
    ),
    # ENUM_DST2 — Direct Space-Time, 2nd order (Lax-Wendroff)
    TracerAdvectionScheme(
        "dst2", 20, halo=1, needs_dt=True, adams_bashforth=False,
        reads_physical_face_mask=False, reads_cell_geometry=False,
        single_rhs=True,
    ),
    # ENUM_DST3 — Direct Space-Time, 3rd order
    TracerAdvectionScheme(
        "dst3", 30, halo=2, needs_dt=True, adams_bashforth=False,
        reads_physical_face_mask=False, reads_cell_geometry=False,
        single_rhs=True,
    ),
    # ENUM_DST3_FLUX_LIMIT — DST3 + Sweby flux limiter
    TracerAdvectionScheme(
        "dst3_fl", 33, halo=2, needs_dt=True, adams_bashforth=False,
        reads_physical_face_mask=False, reads_cell_geometry=False,
        single_rhs=True,
    ),
    # ENUM_FLUX_LIMIT — Superbee flux limiter
    TracerAdvectionScheme(
        "fluxlimit", 77, halo=2, needs_dt=True, adams_bashforth=False,
        reads_physical_face_mask=False, reads_cell_geometry=False,
        single_rhs=True,
    ),
    # ENUM_OS7MP — 7th-order one-step, monotonicity preserving
    TracerAdvectionScheme(
        "os7mp", 7, halo=4, needs_dt=True, adams_bashforth=False,
        reads_physical_face_mask=False, reads_cell_geometry=False,
        single_rhs=True,
    ),
    # ENUM_PPM_NULL_LIMIT — piecewise parabolic, no limiter
    TracerAdvectionScheme(
        "ppm_null", 40, halo=3, needs_dt=True, adams_bashforth=False,
        reads_physical_face_mask=False, reads_cell_geometry=True,
        single_rhs=False,
    ),
    # ENUM_PPM_MONO_LIMIT — piecewise parabolic, monotone limiter
    TracerAdvectionScheme(
        "ppm_mono", 41, halo=3, needs_dt=True, adams_bashforth=False,
        reads_physical_face_mask=False, reads_cell_geometry=True,
        single_rhs=False,
    ),
    # ENUM_PPM_WENO_LIMIT — piecewise parabolic, WENO-weighted limiter
    TracerAdvectionScheme(
        "ppm_weno", 42, halo=4, needs_dt=True, adams_bashforth=False,
        reads_physical_face_mask=False, reads_cell_geometry=True,
        single_rhs=False,
    ),
    # ENUM_PQM_NULL_LIMIT — piecewise quartic, no limiter
    TracerAdvectionScheme(
        "pqm_null", 50, halo=4, needs_dt=True, adams_bashforth=False,
        reads_physical_face_mask=False, reads_cell_geometry=True,
        single_rhs=False,
    ),
    # ENUM_PQM_MONO_LIMIT — piecewise quartic, monotone limiter
    TracerAdvectionScheme(
        "pqm_mono", 51, halo=4, needs_dt=True, adams_bashforth=False,
        reads_physical_face_mask=False, reads_cell_geometry=True,
        single_rhs=False,
    ),
    # ENUM_PQM_WENO_LIMIT — piecewise quartic, WENO-weighted limiter
    TracerAdvectionScheme(
        "pqm_weno", 52, halo=4, needs_dt=True, adams_bashforth=False,
        reads_physical_face_mask=False, reads_cell_geometry=True,
        single_rhs=False,
    ),
)
#: The declaration of each scheme by configuration value.
TRACER_SCHEME_BY_NAME = {scheme.name: scheme for scheme in TRACER_ADVECTION_SCHEMES}

#: Configuration value of every declared scheme (member ``NAME`` holds ``"name"``).
TracerFluxScheme = StrEnum(
    "TracerFluxScheme",
    [(scheme.name.upper(), scheme.name) for scheme in TRACER_ADVECTION_SCHEMES],
    module=__name__,
)


def tracer_scheme_spec(scheme: str) -> TracerAdvectionScheme:
    """Look up either a configuration string or a TracerFluxScheme member."""
    # Dynamo compares dictionary keys by concrete type; the table stores plain strings.
    return TRACER_SCHEME_BY_NAME[str(scheme)]


#: Horizontal schemes of the explicit face-flux paths (``tempAdvScheme``).
HORIZONTAL_TRACER_SCHEMES = (
    TracerFluxScheme.UPWIND,
    TracerFluxScheme.CENTERED,
    TracerFluxScheme.UPWIND3,
    TracerFluxScheme.CENTERED4,
    TracerFluxScheme.DST2,
    TracerFluxScheme.DST3,
    TracerFluxScheme.DST3_FL,
    TracerFluxScheme.FLUXLIMIT,
    TracerFluxScheme.OS7MP,
    TracerFluxScheme.PPM_NULL,
    TracerFluxScheme.PPM_MONO,
    TracerFluxScheme.PPM_WENO,
    TracerFluxScheme.PQM_NULL,
    TracerFluxScheme.PQM_MONO,
    TracerFluxScheme.PQM_WENO,
)
#: Explicit vertical (R) schemes (``tempVertAdvScheme``).
VERTICAL_TRACER_SCHEMES = (
    TracerFluxScheme.UPWIND,
    TracerFluxScheme.CENTERED,
    TracerFluxScheme.UPWIND3,
    TracerFluxScheme.CENTERED4,
    TracerFluxScheme.DST2,
    TracerFluxScheme.DST3,
    TracerFluxScheme.DST3_FL,
    TracerFluxScheme.FLUXLIMIT,
    TracerFluxScheme.OS7MP,
    TracerFluxScheme.PPM_NULL,
    TracerFluxScheme.PPM_MONO,
    TracerFluxScheme.PPM_WENO,
    TracerFluxScheme.PQM_NULL,
    TracerFluxScheme.PQM_MONO,
    TracerFluxScheme.PQM_WENO,
)


#: Horizontal schemes whose face stencil reaches beyond the two cells next to
#: the face (declared ``halo`` above 1).  Every layout adapter (region,
#: six-face, MIT tile) slices the same stencil for them and calls
#: :func:`horizontal_stencil_flux`; the other declared horizontal schemes have
#: their own branch in each adapter.
HORIZONTAL_STENCIL_SCHEMES = tuple(
    scheme for scheme in HORIZONTAL_TRACER_SCHEMES
    if tracer_scheme_spec(scheme).halo > 1
)


def horizontal_stencil_flux(
    scheme: TracerFluxScheme,
    T_left: torch.Tensor,
    T_right: torch.Tensor,
    cells: tuple[torch.Tensor, ...],
    face_masks: tuple[torch.Tensor, ...],
    u_trans: torch.Tensor,
    u_cfl: torch.Tensor,
    *,
    physical_face_masks: tuple[torch.Tensor, ...] | None,
    cell_inputs: StencilCellInputs | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-order upwind and ``scheme`` transport flux on the same faces.

    With ``h`` the declared ``halo`` of ``scheme``: ``cells`` are the
    face-aligned stencils ``T(i-h) .. T(i+h-1)`` and ``face_masks`` the face
    masks ``maskLoc(i-h+1) .. maskLoc(i+h-1)``, with the face transport
    ``u_trans`` and the face Courant number ``|u * dt / dxC|``, all pre-haloed
    by the caller.  ``T_left``/``T_right`` are the upwind donors the caller
    splices onto uncoupled edge faces.  ``physical_face_masks`` are the
    physical face masks ``maskW`` on the faces of ``face_masks``, read only by
    a scheme that declares ``reads_physical_face_mask``; an adapter whose
    ``maskLoc`` carries ``maskInW`` fetches them only for such a scheme and
    passes ``None`` otherwise.  ``cell_inputs`` are the cell masks, cell widths,
    face spacings and face velocity on the same stencil, read only by a scheme
    that declares ``reads_cell_geometry``; ``None`` otherwise.

    This is also the single accelerator consumer boundary for the ``dst3_cuda``
    native kernel: one wiring point serves every layout adapter's DST3 /
    DST3-FL call, and the dispatcher's eager fallback calls the scheme's math
    kernel directly.
    """
    if scheme in (TracerFluxScheme.DST3, TracerFluxScheme.DST3_FL):
        Tm2, Tm1, T0, Tp1 = cells
        mask_im1, mask_i, mask_ip1 = face_masks
        from locomgcm.accel import (
            consumer_snapshot,
            in_accelerated_dispatch,
            requested,
        )

        if not in_accelerated_dispatch():
            with consumer_snapshot():
                if requested(("dst3_cuda",)):
                    from ._dst3_cuda import dispatcher

                    return dispatcher.dst3_transport_flux_values_cuda(
                        T_left,
                        T_right,
                        Tm2,
                        Tm1,
                        T0,
                        Tp1,
                        mask_im1,
                        mask_i,
                        mask_ip1,
                        u_trans,
                        u_cfl,
                        with_limiter=scheme == TracerFluxScheme.DST3_FL,
                    )
        if scheme == TracerFluxScheme.DST3:
            return dst3_transport_flux(
                T_left, T_right, Tm2, Tm1, T0, Tp1,
                mask_im1, mask_i, mask_ip1, u_trans, u_cfl,
            )
        return dst3fl_transport_flux(
            T_left, T_right, Tm2, Tm1, T0, Tp1,
            mask_im1, mask_i, mask_ip1, u_trans, u_cfl,
        )
    if scheme == TracerFluxScheme.UPWIND3:
        return upwind3_transport_flux(T_left, T_right, *cells, *face_masks, u_trans)
    if scheme == TracerFluxScheme.CENTERED4:
        wet_im1, _wet_i, wet_ip1 = physical_face_masks
        return centered4_transport_flux(
            T_left, T_right, *cells, *face_masks, u_trans, wet_im1, wet_ip1,
        )
    if scheme == TracerFluxScheme.FLUXLIMIT:
        return fluxlimit_transport_flux(T_left, T_right, *cells, *face_masks, u_trans, u_cfl)
    if scheme == TracerFluxScheme.OS7MP:
        return os7mp_transport_flux(T_left, T_right, cells, face_masks, u_trans, u_cfl)
    if scheme == TracerFluxScheme.PPM_NULL:
        return ppm_transport_flux("null", T_left, T_right, cells, u_trans, cell_inputs)
    if scheme == TracerFluxScheme.PPM_MONO:
        return ppm_transport_flux("mono", T_left, T_right, cells, u_trans, cell_inputs)
    if scheme == TracerFluxScheme.PPM_WENO:
        return ppm_transport_flux("weno", T_left, T_right, cells, u_trans, cell_inputs)
    if scheme == TracerFluxScheme.PQM_NULL:
        return pqm_transport_flux("null", T_left, T_right, cells, u_trans, cell_inputs)
    if scheme == TracerFluxScheme.PQM_MONO:
        return pqm_transport_flux("mono", T_left, T_right, cells, u_trans, cell_inputs)
    if scheme == TracerFluxScheme.PQM_WENO:
        return pqm_transport_flux("weno", T_left, T_right, cells, u_trans, cell_inputs)
    raise ValueError(f"no face stencil flux for scheme={scheme!r}")


def horizontal_two_point_flux(
    scheme: TracerFluxScheme,
    T_left: torch.Tensor,
    T_right: torch.Tensor,
    transport: torch.Tensor,
    velocity: torch.Tensor,
    cfl: torch.Tensor | None,
) -> torch.Tensor:
    """Face flux of a horizontal scheme whose stencil is the two cells next to the face.

    The declared ``halo`` of these schemes is 1.  ``T_left``/``T_right`` are
    the cells before and after the face, ``transport`` multiplies the face
    value, ``velocity`` picks the donor and ``cfl`` (the face Courant number,
    ``None`` for a scheme with ``needs_dt`` false) is read only by the schemes
    that declare ``needs_dt``.
    """
    if scheme == TracerFluxScheme.UPWIND:
        return upwind_face_flux(T_left, T_right, transport, velocity)
    if scheme == TracerFluxScheme.CENTERED:
        return centered_face_flux(T_left, T_right, transport)
    if scheme == TracerFluxScheme.DST2:
        return dst2_face_flux(T_left, T_right, transport, cfl)
    raise ValueError(f"no two-point face flux for scheme={scheme!r}")


#: Explicit vertical schemes whose kernel reads the interface VELOCITY for its
#: Courant number, with ``recip_drC`` and ``dt``.
VERTICAL_CFL_SCHEMES = tuple(
    scheme for scheme in VERTICAL_TRACER_SCHEMES if tracer_scheme_spec(scheme).needs_dt
)
#: Courant-number schemes whose kernel also weights the flux by that velocity
#: instead of the transport, so a transport-space caller converts the result.
# ponytail: 30 and 33 still take one interface velocity for both; moving them to
# the transport/velocity split of 77, 20 and 7 changes the rounding of their
# existing rows and empties this tuple.
VERTICAL_VELOCITY_FLUX_SCHEMES = (TracerFluxScheme.DST3, TracerFluxScheme.DST3_FL)


def vertical_tracer_flux(
    scheme: TracerFluxScheme | str,
    tracer: torch.Tensor,
    transport: torch.Tensor | None,
    *,
    velocity: torch.Tensor | None,
    recip_drC: torch.Tensor | None,
    dt: float | None,
    mask: torch.Tensor | None,
    layer_thickness: torch.Tensor | None,
    recip_center_spacing: torch.Tensor | None,
) -> torch.Tensor:
    """Interface advective flux of ``scheme`` (DOWNWARD-positive, ``[batch, nz+1, ny, nx]``).

    ``transport`` is the interface volume transport (MITgcm ``-rTrans``), or the
    interface velocity in a velocity-space caller, that the flux is linear in;
    every scheme outside :data:`VERTICAL_VELOCITY_FLUX_SCHEMES` reads it.
    ``velocity`` is the interface velocity (MITgcm ``-wFld``) of the Courant
    number, read with ``recip_drC``/``dt`` by :data:`VERTICAL_CFL_SCHEMES` only;
    the schemes of :data:`VERTICAL_VELOCITY_FLUX_SCHEMES` also weight their flux
    by it.  ``mask`` is the cell wet mask ``maskC`` or ``None``.
    ``layer_thickness`` (``drF``, ``[nz]``) and ``recip_center_spacing``
    (MITgcm ``recip_drC(1..Nr+1)``, ``[nz+1]``, surface and bottom included) are
    read only by a scheme that declares ``reads_cell_geometry``; ``None``
    otherwise.
    """
    if scheme == TracerFluxScheme.UPWIND:
        return vertical_upwind_flux(tracer, transport, mask=mask)
    if scheme == TracerFluxScheme.CENTERED:
        return vertical_centered_flux(tracer, transport, mask=mask)
    if scheme == TracerFluxScheme.DST3:
        return vertical_advective_flux_dst3(
            tracer, velocity, recip_drC=recip_drC, dt=dt, mask=mask,
        )
    if scheme == TracerFluxScheme.DST3_FL:
        return vertical_advective_flux_dst3fl(
            tracer, velocity, recip_drC=recip_drC, dt=dt, mask=mask,
        )
    if scheme == TracerFluxScheme.UPWIND3:
        return vertical_upwind3_flux(tracer, transport, mask=mask)
    if scheme == TracerFluxScheme.CENTERED4:
        return vertical_centered4_flux(tracer, transport, mask=mask)
    if scheme == TracerFluxScheme.DST2:
        return vertical_dst2_flux(
            tracer, transport, velocity, recip_drC=recip_drC, dt=dt, mask=mask,
        )
    if scheme == TracerFluxScheme.OS7MP:
        return vertical_os7mp_flux(
            tracer, transport, velocity, recip_drC=recip_drC, dt=dt, mask=mask,
        )
    if scheme == TracerFluxScheme.FLUXLIMIT:
        return vertical_fluxlimit_flux(
            tracer, transport, velocity, recip_drC=recip_drC, dt=dt, mask=mask,
        )
    if scheme == TracerFluxScheme.PPM_NULL:
        return vertical_ppm_flux(
            "null", tracer, transport, velocity, layer_thickness=layer_thickness,
            dt=dt, mask=mask, halo=tracer_scheme_spec(scheme).halo,
        )
    if scheme == TracerFluxScheme.PPM_MONO:
        return vertical_ppm_flux(
            "mono", tracer, transport, velocity, layer_thickness=layer_thickness,
            dt=dt, mask=mask, halo=tracer_scheme_spec(scheme).halo,
        )
    if scheme == TracerFluxScheme.PPM_WENO:
        return vertical_ppm_flux(
            "weno", tracer, transport, velocity, layer_thickness=layer_thickness,
            dt=dt, mask=mask, halo=tracer_scheme_spec(scheme).halo,
        )
    if scheme == TracerFluxScheme.PQM_NULL:
        return vertical_pqm_flux(
            "null", tracer, transport, velocity, recip_center_spacing=recip_center_spacing,
            layer_thickness=layer_thickness, dt=dt, mask=mask,
            halo=tracer_scheme_spec(scheme).halo,
        )
    if scheme == TracerFluxScheme.PQM_MONO:
        return vertical_pqm_flux(
            "mono", tracer, transport, velocity, recip_center_spacing=recip_center_spacing,
            layer_thickness=layer_thickness, dt=dt, mask=mask,
            halo=tracer_scheme_spec(scheme).halo,
        )
    if scheme == TracerFluxScheme.PQM_WENO:
        return vertical_pqm_flux(
            "weno", tracer, transport, velocity, recip_center_spacing=recip_center_spacing,
            layer_thickness=layer_thickness, dt=dt, mask=mask,
            halo=tracer_scheme_spec(scheme).halo,
        )
    raise ValueError(f"unsupported vertical tracer advection scheme: {scheme!s}")


def interface_recip_drC(grid: Any, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """MITgcm ``recip_drC`` on the interior interfaces ``1 .. nz-1``.

    ``1 / diff(center_depth)``: the center-to-center spacing the CFL schemes'
    Courant number divides by (not the layer thickness ``drF``).
    """
    return 1.0 / torch.diff(grid.center_depth.to(dtype=dtype, device=device))


_X_LEFT_EXTENT = HaloExtentV1(
    west_left_extent=1,
    east_right_extent=0,
    south_bottom_extent=0,
    north_top_extent=0,
)
_X_RIGHT_EXTENT = HaloExtentV1(
    west_left_extent=0,
    east_right_extent=1,
    south_bottom_extent=0,
    north_top_extent=0,
)
_Y_BOTTOM_EXTENT = HaloExtentV1(
    west_left_extent=0,
    east_right_extent=0,
    south_bottom_extent=1,
    north_top_extent=0,
)
_Y_TOP_EXTENT = HaloExtentV1(
    west_left_extent=0,
    east_right_extent=0,
    south_bottom_extent=0,
    north_top_extent=1,
)
#: Halo of the UNIQUE faces 0..n-1 of a face mask (see :func:`_dst3_face_mask_halo`):
#: one face before face 0 and two after face n-1, i.e. faces -1 and n, n+1.
def _cell_stencil_extent(axis: Literal["x", "y"], halo: int) -> HaloExtentV1:
    """Cell halo of a face stencil that reads ``halo`` cells on each side, along ``axis``."""
    if axis == "x":
        return HaloExtentV1(
            west_left_extent=halo, east_right_extent=halo,
            south_bottom_extent=0, north_top_extent=0,
        )
    return HaloExtentV1(
        west_left_extent=0, east_right_extent=0,
        south_bottom_extent=halo, north_top_extent=halo,
    )


def _face_mask_extent(axis: Literal["x", "y"], halo: int) -> HaloExtentV1:
    """Ghosts of the unique faces ``0 .. n-1`` that give faces ``-(halo-1) .. n+halo-1``."""
    if axis == "x":
        return HaloExtentV1(
            west_left_extent=halo - 1, east_right_extent=halo,
            south_bottom_extent=0, north_top_extent=0,
        )
    return HaloExtentV1(
        west_left_extent=0, east_right_extent=0,
        south_bottom_extent=halo - 1, north_top_extent=halo,
    )


def _require_stencil_width(
    grid: StructuredCGrid, extent: int, *, axis: str, halo: int
) -> bool:
    """Check that the local extent can support a face stencil of ``2*halo`` cells.

    A tile at least ``2*halo`` cells wide returns True.  A patch without distributed
    neighbours that is one cell wide returns False so the caller can use
    first-order upwind fluxes: the field is uniform along ``axis`` there, so
    upwind equals every stencil scheme.  Two or three cells wide, the stencil
    would read cells its halo repeats, which this adapter does not implement,
    so the configuration is refused (MITgcm itself accepts it whenever
    ``sNx >= OLx``).
    A narrow distributed tile raises before any exchange in this advection pass:
    skipping its wide-halo exchanges would leave neighbouring ranks waiting for
    unmatched traffic. The launcher propagates the rank-local failure.

    Normal distributed layouts are checked earlier by
    ``distributed.layout.require_minimum_tile_extent``. This check also covers
    entry points that construct the layout through another route.
    """
    min_extent = 2 * halo
    if extent >= min_extent:
        return True
    lower, upper = ("west", "east") if axis == "x" else ("south", "north")
    if not (_has_distributed_side(grid, lower) or _has_distributed_side(grid, upper)):
        if extent == 1:
            return False
        raise ValueError(
            f"a {min_extent}-cell tracer advection stencil needs at least "
            f"{min_extent} cells along {axis}, or exactly 1, but this "
            f"grid has {extent}; use a two-point scheme or a wider domain."
        )
    raise RuntimeError(
        f"a {min_extent}-cell tracer advection stencil needs at least {min_extent} cells along "
        f"{axis} on every tile, but this rank owns {extent}. A tile this narrow "
        f"skips the two wide-halo exchanges of the {axis} pass while its "
        f"neighbours post them, which deadlocks the run. Reduce "
        f"{'nPx' if axis == 'x' else 'nPy'} so that "
        f"{'global_nx // nPx' if axis == 'x' else 'global_ny // nPy'} >= "
        f"{min_extent}."
    )


def _has_distributed_side(grid: StructuredCGrid, side: str) -> bool:
    neighbours = getattr(grid, "distributed_neighbors", None)
    if neighbours is None:
        neighbours = getattr(grid, "distributed_neighbours", None)
    if not neighbours or not hasattr(neighbours, "get"):
        return False
    return neighbours.get(side) is not None


# ---------------------------------------------------------------------------
# Seam semantics of the DST3 pass (numerics rule 11).  One source for the eager
# adapters below and the fused orchestration in
# ``dynamics/_tracer_adv_fused/dispatcher.py``.
# ---------------------------------------------------------------------------


def _dst3_edge_couples(grid: StructuredCGrid, side: str) -> bool:
    """Whether the domain-edge face on ``side`` couples two cells.

    ``grid.edge_face_couples(axis) OR distributed-neighbour`` (numerics rule 11):
    a periodic axis wraps (MITgcm fills the theta and maskW/S halos by EXCH,
    ``do_fields_blocking_exchanges.F:61-62``, on a doubly periodic process
    topology), and a distributed side reads the neighbour rank.  Such a face is
    an interior face of the global domain and takes the full DST3 stencil, as
    ``gad_dst3fl_adv_x.F:47-56`` does for every face ``1-OLx+2 .. sNx+OLx-1``.
    Only a wall or open edge keeps the first-order upwind fallback.
    """
    axis = "x" if side in ("west", "east") else "y"
    return grid.edge_face_couples(axis) or _has_distributed_side(grid, side)


def _dst3_halo_fill(grid: StructuredCGrid, axis: Literal["x", "y"]) -> FillSpec:
    """Ghost fill for the DST3 halos along ``axis`` under the grid's own topology.

    WRAP on a periodic axis; REPLICATE on a wall or open axis, the fill this pass
    has always used there.  A distributed side takes the neighbour strip either
    way (``exchange``).
    """
    return FillSpec.WRAP if grid.edge_face_couples(axis) else FillSpec.REPLICATE


def _dst3_splice_range(grid: StructuredCGrid, axis: Literal["x", "y"]) -> tuple[int, int]:
    """Faces ``[start, stop)`` that keep the DST3 value; the rest are upwind."""
    n = grid.nx if axis == "x" else grid.ny
    lower, upper = ("west", "east") if axis == "x" else ("south", "north")
    start = 0 if _dst3_edge_couples(grid, lower) else 1
    stop = n + 1 if _dst3_edge_couples(grid, upper) else n
    return start, stop


def _dst3_face_mask_halo(
    face_mask: torch.Tensor, grid: StructuredCGrid, axis: Literal["x", "y"], *, halo: int
) -> torch.Tensor:
    """Face mask on faces ``-(halo-1) .. n+halo-1`` (length ``n+2*halo-1``) along ``axis``.

    Faces ``0 .. n`` are the stored values.  The outer ghosts come from an
    exchange of the ``n`` UNIQUE faces ``0 .. n-1``, which carries no duplicate
    seam face: a periodic tile wraps them, so face ``-1`` is face ``n-1`` and face
    ``n+1`` is face ``1`` (MITgcm ``maskW(0) = maskW(sNx)`` after EXCH), and a
    distributed side receives the neighbour's faces.
    Exchanging the full ``n+1`` array would give the same ghosts on a periodic
    axis (``exchange`` skips the duplicate seam face there); the unique-face exchange is kept because it also applies
    on an uncoupled edge, where the outer ghost is read only by the
    upwind-spliced edge face.
    """
    fill = _dst3_halo_fill(grid, axis)
    if axis == "x":
        n = grid.nx
        wide = exchange(
            face_mask[..., :n],
            grid=grid,
            location="U",
            extent=_face_mask_extent("x", halo),
            fill=fill,
            constant=True,
        )                                   # faces -(halo-1) .. n+halo-1
        # Face n is the stored face: the exchanged ghost there equals it on a
        # coupled edge and is the replicated face n-1 on an uncoupled one.
        wide[..., n + halo - 1] = face_mask[..., n]
        return wide
    n = grid.ny
    wide = exchange(
        face_mask[..., :n, :],
        grid=grid,
        location="V",
        extent=_face_mask_extent("y", halo),
        fill=fill,
        constant=True,
    )
    wide[..., n + halo - 1, :] = face_mask[..., n, :]
    return wide


def _stencil_cell_inputs(
    tracer: torch.Tensor,
    velocity: torch.Tensor,
    grid: StructuredCGrid,
    axis: Literal["x", "y"],
    *,
    halo: int,
    dt: float,
) -> StencilCellInputs:
    """Cell masks, cell widths and face spacings on the stencil of every face along ``axis``.

    ``maskC = hFacC > 0``, the row mask MITgcm copies into PPM and PQM
    (``gad_ppm_adv_x.F``: ``mloc(ix) = maskC(ix,iy,kk,bi,bj)``); its ghosts
    follow the grid topology, zero on a wall or open edge, where no wet cell
    lies outside the domain.  The cell widths ``dxF``/``dyF`` and the reciprocal
    face spacings ``1/dxC``/``1/dyC`` take the tracer halo fill.
    """
    dtype, device = tracer.dtype, tracer.device
    if grid.masks.hFacC is not None:
        maskC = (grid.masks.hFacC.to(dtype=dtype, device=device) > 0).to(dtype)
    else:
        maskC = torch.ones_like(tracer)
    extent = _cell_stencil_extent(axis, halo)
    mask_fill = FillSpec.WRAP if grid.edge_face_couples(axis) else FillSpec.ZERO
    mask_wide = exchange(
        maskC, grid=grid, location="T", extent=extent, fill=mask_fill, constant=True,
    )
    if axis == "x":
        width = grid.metrics.dxF_for(dtype, device)
        recip_spacing = 1.0 / grid.metrics.dxC_for(dtype, device)
    else:
        width = grid.metrics.dyF_for(dtype, device)
        recip_spacing = 1.0 / grid.metrics.dyC_for(dtype, device)
    width_wide = exchange(
        width, grid=grid, location="T", extent=extent, fill=_dst3_halo_fill(grid, axis),
    )
    spacing_wide = _dst3_face_mask_halo(recip_spacing, grid, axis, halo=halo)
    dim, n = (-1, grid.nx) if axis == "x" else (-2, grid.ny)
    return StencilCellInputs(
        cell_masks=tuple(mask_wide.narrow(dim, k, n + 1) for k in range(2 * halo)),
        cell_widths=tuple(width_wide.narrow(dim, k, n + 1) for k in range(2 * halo)),
        recip_face_spacings=tuple(
            spacing_wide.narrow(dim, k, n + 1) for k in range(2 * halo - 1)
        ),
        velocity=velocity,
        dt=dt,
    )


def _recip_face_spacing(grid: StructuredCGrid, axis: Literal["x", "y"], tracer: torch.Tensor) -> torch.Tensor:
    """Region ``1/dxC`` on the U faces (``axis="x"``) or ``1/dyC`` on the V faces, in the
    dtype and device of ``tracer`` and viewed over its leading dimensions."""
    lead = [1] * max(tracer.ndim - 2, 0)
    if axis == "x":
        dxC = grid.metrics.dxC_for(tracer.dtype, tracer.device)
        return (1.0 / dxC).view(*lead, grid.ny, grid.nx + 1)
    dyC = grid.metrics.dyC_for(tracer.dtype, tracer.device)
    return (1.0 / dyC).view(*lead, grid.ny + 1, grid.nx)


def _stencil_face_flux_x(
    tracer: torch.Tensor,
    u_face: torch.Tensor,
    u_trans: torch.Tensor,
    grid: StructuredCGrid,
    *,
    dt: float,
    scheme: TracerFluxScheme,
) -> torch.Tensor:
    """Compute MITgcm-equivalent transport-form flux ``uT(i,j) = af`` on U faces.

    Returns a tensor of shape ``[..., ny, nx+1]`` (full u-face array).  Every
    face that couples two cells takes the DST3 value, including the edge faces
    of a periodic axis or of a distributed seam (:func:`_dst3_edge_couples`); an
    edge face on a wall or open axis falls back to first-order upwind transport
    flux.  Mirrors GAD_DST3FL_ADV_X (gad_dst3fl_adv_x.F:42-99) including the
    transport-form output shape.

    Parameters
    ----------
    tracer    : [..., nz, ny, nx]   cell-centered tracer
    u_face    : [..., nz, ny, nx+1] velocity at u face (uFld in MIT)
    u_trans   : [..., nz, ny, nx+1] transport at u face (uTrans = u·dyG·drF·hFacW)
    dt        : float                                Δt for CFL
    scheme    : TracerFluxScheme    one of ``HORIZONTAL_STENCIL_SCHEMES``

    Notes
    -----
    The output is **transport-form flux** (already includes the velocity
    branching), NOT a face tracer value.  This matches what
    ``gad_advection.F`` consumes as ``af`` for the multi-dim splitting.
    """
    nx = grid.nx
    spec = tracer_scheme_spec(scheme)
    halo = spec.halo
    # Width backstop first — before ANY exchange of this pass: on a
    # distributed short tile it raises with zero traffic posted; on a single
    # patch one cell wide it returns False and the upwind fallback below still
    # runs (its two edge exchanges are local fills, cheap and needed either
    # way); two or three cells wide it raises.
    stencil_fits = _require_stencil_width(grid, nx, axis="x", halo=halo)
    # Reproduce MITgcm ``maskLocW = _maskW * maskInW`` (gad_advection.F:321-323):
    #   _maskW  = (hFacW > 0)  — physical wet/dry
    #   maskInW = OBCS interior mask, post-zero of OB strip + halo W-faces
    #             per obcs_init_fixed.F:295-330 (E/W) and :347-367 (N/S
    #             tangential strips).
    # Without the maskInW factor the kernel produces non-MIT slopes at
    # OB-strip / halo-equivalent W faces; production runs hide it via the
    # update-side ``maskInC=0`` zeroing on those cells, but the kernel
    # definition is incomplete without explicit maskInW.
    if grid.masks.hFacW is not None and tracer.ndim >= 4:
        hFacW = grid.masks.hFacW.to(dtype=tracer.dtype, device=tracer.device)
        maskW = (hFacW > 0).to(dtype=tracer.dtype)
        maskLocW = maskW
        if grid.masks.maskInW is not None:
            maskInW = grid.masks.maskInW.to(dtype=tracer.dtype, device=tracer.device)
            maskLocW = maskW * maskInW  # broadcasts [ny, nx+1] over [nz, ny, nx+1]
    else:
        maskW = maskLocW = torch.ones_like(u_face)

    # First-order upwind fallback flux for the uncoupled edge faces and for the
    # full-array initialisation (matching MIT's loop that initialises
    # uT(1-OLx,j)=uT(2-OLx,j)=uT(sNx+OLx,j)=0 then fills interior).
    # Upwind transport flux on face i: 0.5*(uTrans+|uTrans|)*T(i-1) + 0.5*(uTrans-|uTrans|)*T(i)
    # Build T_left/T_right arrays of length nx+1 by padding edges.  Every halo
    # of this pass follows the grid's own topology (numerics rule 11): the
    # periodic wrap or the neighbour rank on a coupled edge, and on a wall or
    # open edge the nearest cell, T(-1) := T(0) (the mask handles the wall).
    fill = _dst3_halo_fill(grid, "x")
    T_left_full = exchange(
        tracer,
        grid=grid,
        location="T",
        extent=_X_LEFT_EXTENT,
        fill=fill,
    )        # [..., ny, nx+1]
    T_right_full = exchange(
        tracer,
        grid=grid,
        location="T",
        extent=_X_RIGHT_EXTENT,
        fill=fill,
    )      # [..., ny, nx+1]

    # DST3 region: faces [1 .. nx-1] plus each edge face that couples two cells
    # (periodic wrap or distributed seam, _dst3_splice_range).  An uncoupled
    # edge face (wall or open) is the physical boundary face: it enters only the
    # boundary cell update, which has maskInC=0 in OBCS configurations, so a
    # first-order upwind fallback there is bitwise-equivalent to MIT's
    # halo-zero handling.
    #
    # The 4-point stencil needs T(i-2..i+1) at face i.  On an uncoupled edge,
    # face 1 needs T(-1) and face nx-1 needs T(nx), which do not exist.  The
    # nearest in-domain cell pads both, making Rjm[face 1] = (T(0)-T(0))·mask
    # = 0 and Rjp[face nx-1] = (T(nx-1)-T(nx-1))·mask = 0.  This **mirrors
    # MIT's masked behavior** — at face 1 (= MIT face 2 with PyTorch face index
    # = MIT face index − 1 in a no-halo mapping), MIT's maskInW(MIT face 1)
    # is zeroed by ``obcs_init_fixed.F:320-324`` for the west OB, so MIT's
    # Rjm at MIT face 2 is also exactly zero; symmetric for east OB at
    # face nx-1 ↔ MIT face sNx, where ``obcs_init_fixed.F:301-306`` zeros
    # maskInW(sNx+1) → MIT Rjp = 0.  Both ends therefore degenerate to the
    # same DST3-FL configuration on both implementations.
    # On the DST3 path horizontal_stencil_flux recomputes the very same
    # pos/neg/upwind triple internally and rebinds upwind_flux, so building it
    # here only pays off on the fallback branch.  The width verdict itself was
    # taken at entry, before any exchange.
    if not stencil_fits:
        # single patch one cell wide: uniform along the axis, upwind equals the stencil
        pos_full = 0.5 * (u_trans + u_trans.abs())
        neg_full = 0.5 * (u_trans - u_trans.abs())
        return pos_full * T_left_full + neg_full * T_right_full

    # Slice T(i-2..i+1) for active DST3 face indices.  Exchange the full tracer
    # before slicing.  Passing sliced tensors to a distributed exchange sends
    # the edge of the slice, not the true owned edge, so seam-adjacent faces
    # would read T(i-4)/T(i-2) instead of neighbour T.
    tracer_wide = exchange(
        tracer,
        grid=grid,
        location="T",
        extent=_cell_stencil_extent("x", halo),
        fill=fill,
    )
    # T(i-halo .. i+halo-1) on faces 0 .. nx (cell c sits at wide index c+halo).
    cells = tuple(tracer_wide[..., k:nx + 1 + k] for k in range(2 * halo))

    # Face-centered masks for each R difference (gad_dst3fl_adv_x.F:54-56):
    #   Rjp = (T(i+1)-T(i)) * maskLocW(i+1)
    #   Rj  = (T(i)-T(i-1)) * maskLocW(i)
    #   Rjm = (T(i-1)-T(i-2)) * maskLocW(i-1)
    # maskLocW on faces i-halo+1 .. i+halo-1 for face index i.
    maskLocW_wide = _dst3_face_mask_halo(maskLocW, grid, "x", halo=halo)
    face_masks = tuple(maskLocW_wide[..., k:nx + 1 + k] for k in range(2 * halo - 1))
    physical_face_masks = None
    if spec.reads_physical_face_mask:
        # Physical maskW on the same faces (gad_c4_adv_x.F:72), no maskInW.
        maskW_wide = _dst3_face_mask_halo(maskW, grid, "x", halo=halo)
        physical_face_masks = tuple(
            maskW_wide[..., k:nx + 1 + k] for k in range(2 * halo - 1)
        )

    cell_inputs = None
    if spec.reads_cell_geometry:
        cell_inputs = _stencil_cell_inputs(tracer, u_face, grid, "x", halo=halo, dt=dt)

    u_cfl = face_courant_number(u_face, _recip_face_spacing(grid, "x", tracer), dt)

    # DST3(-FL) reconstruction math lives in the shared geometry-neutral core
    # (Rjp/Rj/Rjm, theta-guard, psi, transport-form flux).  Same core is reused
    # by the cube tile adapter.
    upwind_flux, flux_dst = horizontal_stencil_flux(
        scheme, T_left_full, T_right_full, cells, face_masks, u_trans, u_cfl,
        physical_face_masks=physical_face_masks, cell_inputs=cell_inputs,
    )

    dst_start, dst_stop = _dst3_splice_range(grid, "x")
    if dst_start >= dst_stop:
        return upwind_flux
    # Splice the other way round: flux_dst already holds the DST3 value on
    # [dst_start, dst_stop) — the whole array bar at most one column at each
    # end — so overwrite those edge columns from upwind_flux in place instead of
    # cloning a full field and copying the interior back over it.  Same values,
    # same faces; flux_dst is freshly produced by horizontal_stencil_flux
    # (eager or CUDA) and has no other reader.
    flux_full = flux_dst
    if dst_start > 0:
        flux_full[..., :dst_start] = upwind_flux[..., :dst_start]
    if dst_stop < nx + 1:
        flux_full[..., dst_stop:] = upwind_flux[..., dst_stop:]
    return flux_full


def _stencil_face_flux_y(
    tracer: torch.Tensor,
    v_face: torch.Tensor,
    v_trans: torch.Tensor,
    grid: StructuredCGrid,
    *,
    dt: float,
    scheme: TracerFluxScheme,
) -> torch.Tensor:
    """Y-direction mirror of ``_stencil_face_flux_x``.

    Mirrors GAD_DST3FL_ADV_Y (gad_dst3fl_adv_y.F:38-95).
    Returns shape [..., nz, ny+1, nx]; faces j=0 and j=ny fall back to
    first-order upwind only on a wall or open y edge.
    """
    ny = grid.ny
    spec = tracer_scheme_spec(scheme)
    halo = spec.halo
    # Width backstop first — before ANY exchange of this pass; see the
    # X analog for the call-order contract.
    stencil_fits = _require_stencil_width(grid, ny, axis="y", halo=halo)
    # Reproduce MITgcm ``maskLocS = _maskS * maskInS`` (gad_advection.F:321-323).
    # See the X analog above for why both factors are required.
    if grid.masks.hFacS is not None and tracer.ndim >= 4:
        hFacS = grid.masks.hFacS.to(dtype=tracer.dtype, device=tracer.device)
        maskS = (hFacS > 0).to(dtype=tracer.dtype)
        maskLocS = maskS
        if grid.masks.maskInS is not None:
            maskInS = grid.masks.maskInS.to(dtype=tracer.dtype, device=tracer.device)
            maskLocS = maskS * maskInS  # broadcasts [ny+1, nx] over [nz, ny+1, nx]
    else:
        maskS = maskLocS = torch.ones_like(v_face)

    # Halos follow the grid's own topology; see the X analog.
    fill = _dst3_halo_fill(grid, "y")
    T_bot_full = exchange(
        tracer,
        grid=grid,
        location="T",
        extent=_Y_BOTTOM_EXTENT,
        fill=fill,
    )
    T_top_full = exchange(
        tracer,
        grid=grid,
        location="T",
        extent=_Y_TOP_EXTENT,
        fill=fill,
    )

    # Width verdict taken at entry, before any exchange.
    if not stencil_fits:
        # single patch one cell wide: uniform along the axis, upwind equals the stencil
        pos_full = 0.5 * (v_trans + v_trans.abs())
        neg_full = 0.5 * (v_trans - v_trans.abs())
        return pos_full * T_bot_full + neg_full * T_top_full

    # DST3 region: faces [1 .. ny-1] plus each south/north edge face that
    # couples two cells (periodic wrap or distributed seam).
    tracer_wide = exchange(
        tracer,
        grid=grid,
        location="T",
        extent=_cell_stencil_extent("y", halo),
        fill=fill,
    )
    cells = tuple(tracer_wide[..., k:ny + 1 + k, :] for k in range(2 * halo))

    maskLocS_wide = _dst3_face_mask_halo(maskLocS, grid, "y", halo=halo)
    face_masks = tuple(maskLocS_wide[..., k:ny + 1 + k, :] for k in range(2 * halo - 1))
    physical_face_masks = None
    if spec.reads_physical_face_mask:
        # Physical maskS on the same faces (gad_c4_adv_y.F:72), no maskInS.
        maskS_wide = _dst3_face_mask_halo(maskS, grid, "y", halo=halo)
        physical_face_masks = tuple(
            maskS_wide[..., k:ny + 1 + k, :] for k in range(2 * halo - 1)
        )

    cell_inputs = None
    if spec.reads_cell_geometry:
        cell_inputs = _stencil_cell_inputs(tracer, v_face, grid, "y", halo=halo, dt=dt)

    v_cfl = face_courant_number(v_face, _recip_face_spacing(grid, "y", tracer), dt)

    # Shared geometry-neutral DST3(-FL) core (same as the X adapter).
    upwind_flux, flux_dst = horizontal_stencil_flux(
        scheme, T_bot_full, T_top_full, cells, face_masks, v_trans, v_cfl,
        physical_face_masks=physical_face_masks, cell_inputs=cell_inputs,
    )

    dst_start, dst_stop = _dst3_splice_range(grid, "y")
    if dst_start >= dst_stop:
        return upwind_flux
    # See the X analog: splice the edge rows onto flux_dst in place.
    flux_full = flux_dst
    if dst_start > 0:
        flux_full[..., :dst_start, :] = upwind_flux[..., :dst_start, :]
    if dst_stop < ny + 1:
        flux_full[..., dst_stop:, :] = upwind_flux[..., dst_stop:, :]
    return flux_full


_X_TWO_POINT_EXTENT = HaloExtentV1(
    west_left_extent=1,
    east_right_extent=1,
    south_bottom_extent=0,
    north_top_extent=0,
)
_Y_TWO_POINT_EXTENT = HaloExtentV1(
    west_left_extent=0,
    east_right_extent=0,
    south_bottom_extent=1,
    north_top_extent=1,
)


def _two_point_face_flux_x(
    tracer: torch.Tensor,
    transport: torch.Tensor,
    velocity: torch.Tensor,
    grid: StructuredCGrid,
    *,
    scheme: TracerFluxScheme,
    dt: float,
    obc_west_tracer: torch.Tensor | None,
    obc_east_tracer: torch.Tensor | None,
) -> torch.Tensor:
    """Region adapter of :func:`horizontal_two_point_flux` on the U faces ``[..., ny, nx+1]``.

    One 1-cell halo exchange of ``tracer``, the one ``center_to_u_upwind`` and
    ``center_to_u_centered`` post: OBC_VALUE fill with the open-boundary tracer
    values where given, replicate on other non-periodic edges, wrap on a
    periodic axis.  The face Courant number is formed only for a scheme that
    declares ``needs_dt``.
    """
    obc = {}
    obc_west = prepare_obc_tracer(obc_west_tracer, tracer=tracer, side_len=grid.ny, side_name="west")
    obc_east = prepare_obc_tracer(obc_east_tracer, tracer=tracer, side_len=grid.ny, side_name="east")
    if obc_west is not None:
        obc["west"] = obc_west
    if obc_east is not None:
        obc["east"] = obc_east
    tracer_halo = exchange(
        tracer,
        grid=grid,
        location="T",
        extent=_X_TWO_POINT_EXTENT,
        fill=FillSpec.OBC_VALUE,
        obc=obc,
    )
    cfl = (
        face_courant_number(velocity, _recip_face_spacing(grid, "x", tracer), dt)
        if tracer_scheme_spec(scheme).needs_dt
        else None
    )
    return horizontal_two_point_flux(
        scheme, tracer_halo[..., :, :-1], tracer_halo[..., :, 1:], transport, velocity, cfl,
    )


def _two_point_face_flux_y(
    tracer: torch.Tensor,
    transport: torch.Tensor,
    velocity: torch.Tensor,
    grid: StructuredCGrid,
    *,
    scheme: TracerFluxScheme,
    dt: float,
    obc_south_tracer: torch.Tensor | None,
    obc_north_tracer: torch.Tensor | None,
) -> torch.Tensor:
    """Y mirror of :func:`_two_point_face_flux_x` on the V faces ``[..., ny+1, nx]``."""
    obc = {}
    obc_south = prepare_obc_tracer(obc_south_tracer, tracer=tracer, side_len=grid.nx, side_name="south")
    obc_north = prepare_obc_tracer(obc_north_tracer, tracer=tracer, side_len=grid.nx, side_name="north")
    if obc_south is not None:
        obc["south"] = obc_south
    if obc_north is not None:
        obc["north"] = obc_north
    tracer_halo = exchange(
        tracer,
        grid=grid,
        location="T",
        extent=_Y_TWO_POINT_EXTENT,
        fill=FillSpec.OBC_VALUE,
        obc=obc,
    )
    cfl = (
        face_courant_number(velocity, _recip_face_spacing(grid, "y", tracer), dt)
        if tracer_scheme_spec(scheme).needs_dt
        else None
    )
    return horizontal_two_point_flux(
        scheme, tracer_halo[..., :-1, :], tracer_halo[..., 1:, :], transport, velocity, cfl,
    )


def _apply_obcs_upwind_override(
    flux_u: torch.Tensor,
    flux_v: torch.Tensor,
    velocity_u: torch.Tensor,
    velocity_v: torch.Tensor,
    tracer: torch.Tensor,
    grid: StructuredCGrid,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace tracer advective flux at OBC faces with 1st-order upwind.

    MITgcm equivalent: ``OBCS_U1_ADV_TRACER`` in ``obcs_u1_adv_tracer.F``.
    At faces where ``maskInC(i-1) != maskInC(i)`` (the boundary between
    OBC ghost cells and interior cells), the centered-scheme flux is
    replaced with upwind:  ``flux = 0.5*(uTr+|uTr|)*T_left + 0.5*(uTr-|uTr|)*T_right``.

    This prevents the centered scheme from extrapolating tracer values
    across the open boundary, which can create spurious oscillations.
    """
    open_x = grid.open_boundary("x")
    open_y = grid.open_boundary("y")
    if not (open_x or open_y):
        return flux_u, flux_v

    maskInC, _, _ = _compute_interior_mask(
        grid, flux_u.dtype, flux_u.device,
    )

    # hFacW weighting (same as in horizontal_tracer_flux)
    hFacW = None
    hFacS = None
    if grid.masks.hFacW is not None and flux_u.ndim >= 4:
        hFacW = grid.masks.hFacW.to(dtype=flux_u.dtype, device=flux_u.device)
    if grid.masks.hFacS is not None and flux_v.ndim >= 4:
        hFacS = grid.masks.hFacS.to(dtype=flux_v.dtype, device=flux_v.device)

    # --- X-direction: faces where maskInC differs across the face ---
    # face i separates cell (i-1) on left from cell (i) on right
    # Interior faces: indices 1 .. nx-1 (non-periodic)
    nx = grid.nx
    ny = grid.ny
    if open_x:
        # maskInC: [ny, nx]
        # Check maskInC(i-1) != maskInC(i) for i in [1, nx-1]
        mL = maskInC[:, :-1]  # [ny, nx-1] cells 0..nx-2
        mR = maskInC[:, 1:]   # [ny, nx-1] cells 1..nx-1
        ob_face_mask = (mL != mR)  # [ny, nx-1] True at OB faces

        # Compute unconditionally so a device-resident OBC mask never becomes
        # a host branch; torch.where leaves every non-OBC face unchanged.
        T_left = tracer[..., :-1]   # [B, nz, ny, nx-1] cell i-1
        T_right = tracer[..., 1:]   # [B, nz, ny, nx-1] cell i
        u_face = velocity_u[..., 1:nx]  # [B, nz, ny, nx-1] faces 1..nx-1
        upwind_flux = first_order_upwind_flux(u_face, T_left, T_right)
        if hFacW is not None and upwind_flux.ndim >= 4:
            upwind_flux = upwind_flux * hFacW[:, :, 1:nx]

        # ob_face_mask is [ny, nx-1]: broadcast to [1,1,ny,nx-1] or [1,ny,nx-1]
        if flux_u.ndim >= 4:
            mask_expand = ob_face_mask.unsqueeze(0).unsqueeze(0)
        else:
            mask_expand = ob_face_mask.unsqueeze(0)

        flux_u = flux_u.clone()
        flux_u[..., 1:nx] = torch.where(mask_expand, upwind_flux, flux_u[..., 1:nx])

    if open_y:
        mB = maskInC[:-1, :]  # cells 0..ny-2
        mT = maskInC[1:, :]   # cells 1..ny-1
        ob_face_mask = (mB != mT)

        T_bot = tracer[..., :-1, :]
        T_top = tracer[..., 1:, :]
        v_face = velocity_v[..., 1:ny, :]
        upwind_flux = first_order_upwind_flux(v_face, T_bot, T_top)
        if hFacS is not None and upwind_flux.ndim >= 4:
            upwind_flux = upwind_flux * hFacS[:, 1:ny, :]

        if flux_v.ndim >= 4:
            mask_expand = ob_face_mask.unsqueeze(0).unsqueeze(0)
        else:
            mask_expand = ob_face_mask.unsqueeze(0)

        flux_v = flux_v.clone()
        flux_v[..., 1:ny, :] = torch.where(mask_expand, upwind_flux, flux_v[..., 1:ny, :])

    return flux_u, flux_v


def horizontal_tracer_flux(
    tracer: torch.Tensor,
    velocity_u: torch.Tensor,
    velocity_v: torch.Tensor,
    grid: StructuredCGrid,
    *,
    scheme: TracerFluxScheme = TracerFluxScheme.UPWIND,
    obcs_upwind: bool = False,
    dt: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Horizontal tracer flux on U/V faces, weighted by ``hFacW``/``hFacS``.

    ``dt`` is the tracer time step, required only by the schemes that declare
    ``needs_dt`` (``deltaTLev(k)`` in ``gad_calc_rhs.F:280-283,411-419``) for
    their Courant number.  The others ignore it.
    """
    if tracer_scheme_spec(scheme).needs_dt and not dt > 0:
        raise ValueError(
            f"scheme={scheme} needs the tracer time step for its Courant "
            "number (MITgcm deltaTLev); pass dt>0 "
            "(HorizontalTracerAdvection(dt=...))"
        )
    if scheme in HORIZONTAL_STENCIL_SCHEMES:
        # MITgcm never forces multi-dim for DST3 / DST3-FL: gad_init_fixed.F:
        # 126-137 clears tempMultiDimAdvec/saltMultiDimAdvec only for
        # ENUM_CENTERED_2ND / ENUM_UPWIND_3RD / ENUM_CENTERED_4TH.  With
        # multiDimAdvection=.FALSE. the schemes run through the single-RHS path
        # (temp_integrate.F:276-295 sets calcAdvection=.TRUE. and skips
        # GAD_ADVECTION), where GAD_CALC_RHS dispatches GAD_DST3_ADV_X /
        # GAD_DST3FL_ADV_X on the SAME T^n as the Y and R passes
        # (gad_calc_rhs.F:280-283 X, :411-419 Y).
        #
        # GAD_DST3{,FL}_ADV_X already return the transport-form flux ``af`` --
        # the velocity branching is folded into the pos/neg factors inside
        # ``horizontal_stencil_flux``, which consumes the ``u_trans`` built
        # HERE.  The kernel itself never multiplies by hFacW (it only derives
        # the 0/1 ``maskLocW`` from it), so the face weight has to be in
        # ``u_trans`` on entry and must NOT be applied a second time on the way
        # out -- hence the early return below, past the shared hFac multiply.
        u_trans = velocity_u
        v_trans = velocity_v
        if grid.masks.hFacW is not None and velocity_u.ndim >= 4:
            hFacW = grid.masks.hFacW.to(dtype=velocity_u.dtype, device=velocity_u.device)
            u_trans = velocity_u * hFacW
        if grid.masks.hFacS is not None and velocity_v.ndim >= 4:
            hFacS = grid.masks.hFacS.to(dtype=velocity_v.dtype, device=velocity_v.device)
            v_trans = velocity_v * hFacS
        flux_u = _stencil_face_flux_x(
            tracer, velocity_u, u_trans, grid, dt=dt, scheme=scheme,
        )
        flux_v = _stencil_face_flux_y(
            tracer, velocity_v, v_trans, grid, dt=dt, scheme=scheme,
        )
        if obcs_upwind:
            flux_u, flux_v = _apply_obcs_upwind_override(
                flux_u, flux_v, velocity_u, velocity_v, tracer, grid,
            )
        return flux_u, flux_v

    flux_u = _two_point_face_flux_x(
        tracer, velocity_u, velocity_u, grid,
        scheme=scheme, dt=dt, obc_west_tracer=None, obc_east_tracer=None,
    )
    flux_v = _two_point_face_flux_y(
        tracer, velocity_v, velocity_v, grid,
        scheme=scheme, dt=dt, obc_south_tracer=None, obc_north_tracer=None,
    )

    # Weight fluxes by hFacW/hFacS (open face-area fraction).
    # MITgcm: uTrans = u * dyG * drF * hFacW; the dyG*drF factors cancel in
    # the uniform-grid divergence denominator, but hFacW remains.  At partial
    # cell faces the flux must be reduced proportionally to the open area.
    if grid.masks.hFacW is not None and flux_u.ndim >= 4:
        hFacW = grid.masks.hFacW.to(dtype=flux_u.dtype, device=flux_u.device)
        flux_u = flux_u * hFacW
    if grid.masks.hFacS is not None and flux_v.ndim >= 4:
        hFacS = grid.masks.hFacS.to(dtype=flux_v.dtype, device=flux_v.device)
        flux_v = flux_v * hFacS

    # Replace flux at OBC faces with 1st-order upwind
    # (MITgcm: OBCS_U1_ADV_TRACER in gad_calc_rhs.F L296-303)
    if obcs_upwind:
        flux_u, flux_v = _apply_obcs_upwind_override(
            flux_u, flux_v, velocity_u, velocity_v, tracer, grid,
        )

    return flux_u, flux_v


def horizontal_tracer_flux_divergence(
    tracer: torch.Tensor,
    velocity_u: torch.Tensor,
    velocity_v: torch.Tensor,
    grid: StructuredCGrid,
    *,
    scheme: TracerFluxScheme = TracerFluxScheme.UPWIND,
    obcs_upwind: bool = False,
    dt: float = 0.0,
) -> torch.Tensor:
    flux_u, flux_v = horizontal_tracer_flux(
        tracer,
        velocity_u,
        velocity_v,
        grid,
        scheme=scheme,
        obcs_upwind=obcs_upwind,
        dt=dt,
    )
    return tracer_flux_divergence_to_center(flux_u, flux_v, grid)
