"""Horizontal Laplacian diffusion of the centre tracers (MITgcm GAD_DIFF_X/Y).

The two kernels take tracers already haloed one cell along x (``tracer_x``,
``[..., ny, nx+2]``) and along y (``tracer_y``, ``[..., ny+2, nx]``) and return
the centre tendency ``[..., ny, nx]`` in tracer units per second.  Face fractions
``hFacW_x``/``hFacS_y`` sit on the U and V faces, the final ``1/hFacC`` is a
positive reciprocal (zero in dry cells), and every tensor shares the tracer
dtype.  The biharmonic term (diffK4, GAD_GRAD/GAD_DEL2/GAD_BIHARM) is two kernels:
``del2`` from the haloed tracer, then its flux divergence from ``del2`` haloed by
the caller.  :class:`HorizontalTracerDiffusion` reads the grid, builds the halos
and picks the kernel; the kernels never see the grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from locomgcm.geometry import StructuredCGrid
from locomgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange
from locomgcm.geometry.metric_bundle import _positive_reciprocal
from locomgcm.operators import laplacian_center
from locomgcm.state import DiagnosticState, PrognosticState, StateTendency

from .base import TendencyModule, TendencyPhase, TendencyStepDomain
from locomgcm.dynamics._registry import register_dynamics_module


def _horizontal_tracer_diffusion_tendency_from_haloes(
    tracer: torch.Tensor,
    tracer_x: torch.Tensor,
    tracer_y: torch.Tensor,
    hFacC: torch.Tensor,
    hFacW_x: torch.Tensor,
    hFacS_y: torch.Tensor,
    diffusivity: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
) -> torch.Tensor:
    """MIT-aligned hFac-aware horizontal tracer diffusion from haloed inputs."""
    ny = tracer.shape[-2]
    nx = tracer.shape[-1]
    dfx = torch.zeros(
        (*tracer.shape[:-1], nx + 1),
        dtype=tracer.dtype,
        device=tracer.device,
    )
    dfx[...] = (
        diffusivity * hFacW_x[..., 1:nx + 2]
        * (tracer_x[..., :nx + 1] - tracer_x[..., 1:nx + 2]) / dx
    )

    dfy = torch.zeros(
        (*tracer.shape[:-2], ny + 1, nx),
        dtype=tracer.dtype,
        device=tracer.device,
    )
    dfy[...] = (
        diffusivity * hFacS_y[..., 1:ny + 2, :]
        * (tracer_y[..., :ny + 1, :] - tracer_y[..., 1:ny + 2, :]) / dy
    )

    div = (
        (dfx[..., 1:] - dfx[..., :-1]) / dx
        + (dfy[..., 1:, :] - dfy[..., :-1, :]) / dy
    )
    tendency = -div
    recip_hFacC = _positive_reciprocal(hFacC)
    return tendency * recip_hFacC


def horizontal_tracer_diffusion_flux_form(
    tracer_x: torch.Tensor,
    tracer_y: torch.Tensor,
    *,
    dyG_x: torch.Tensor,
    dxG_y: torch.Tensor,
    recip_dxC_x: torch.Tensor,
    recip_dyC_y: torch.Tensor,
    hFacW_x: torch.Tensor,
    hFacS_y: torch.Tensor,
    recip_rA: torch.Tensor,
    recip_hFacC: torch.Tensor,
    diffKh: torch.Tensor | float,
    nx: int,
    ny: int,
    maskInC: torch.Tensor | None = None,
) -> torch.Tensor:
    """Geometry-neutral MIT GAD_DIFF full-flux-form horizontal tracer diffusion.

    Pure function: takes PRE-HALOED tracer + 2-D face metrics and does NO
    internal exchange (the caller's :class:`HaloExchange`-style adapter supplies
    the cross-face / periodic halo).  Reproduces MITgcm ``GAD_DIFF_X/Y`` +
    ``gad_calc_rhs.F`` divergence (``references/MITgcm/pkg/generic_advdiff``)::

        DFx(i,j) = -diffKh * xA * recip_dxC * (T(i,j) - T(i-1,j)),  xA = dyG*drF*hFacW
        gTracer  = -recip_rA * recip_drF * recip_hFacC
                   * [ (DFx(i+1)-DFx(i)) + (DFy(j+1)-DFy(j)) ] * maskInC

    ``drF`` CANCELS (``xA`` carries ``drF``, the divergence carries
    ``recip_drF``) so the 2-D-per-layer form passes only ``dyG``/``dxG`` (no
    ``drF``); the layer index lives in ``hFac*`` + ``recip_hFacC``.  Boussinesq
    ``deepFacC = rhoFacC = 1`` is assumed (production cube).

    Anchors its indexing on :func:`_horizontal_tracer_diffusion_tendency_from_haloes`
    (the region simplified-stencil sibling): ``tracer_x`` is haloed along x so
    array index ``i`` holds cell ``i-1`` and U-face ``i`` (the edge west of cell
    ``i``) uses ``tracer_x[..., i+1] - tracer_x[..., i]``.  Unlike the region
    sibling this is the FULL flux form (carries ``dyG``/``recip_rA``/``recip_hFacC``),
    MIT-faithful on 2-D-varying (cube / partial-cell) metrics where the region
    ``1/dxC^2`` reduction (valid only when ``rA = dyG*dxC`` and ``hFacC == 1``)
    is not.

    Shapes (``L`` = leading face/batch dims, ``z`` = vertical or broadcast 1):

    - ``tracer_x``    ``[*L, nz, ny, nx+2]``   (cells ``-1..nx`` along x)
    - ``tracer_y``    ``[*L, nz, ny+2, nx]``   (cells ``-1..ny`` along y)
    - ``dyG_x``       ``[*L, 1, ny, nx+1]``    (U-face zonal edge, faces 0..nx)
    - ``dxG_y``       ``[*L, 1, ny+1, nx]``    (V-face merid edge, faces 0..ny)
    - ``recip_dxC_x`` ``[*L, 1, ny, nx+1]``
    - ``recip_dyC_y`` ``[*L, 1, ny+1, nx]``
    - ``hFacW_x``     ``[*L, nz, ny, nx+1]``   (U-face hFacW, faces 0..nx)
    - ``hFacS_y``     ``[*L, nz, ny+1, nx]``   (V-face hFacS, faces 0..ny)
    - ``recip_rA``    ``[*L, 1, ny, nx]``
    - ``recip_hFacC`` ``[*L, nz, ny, nx]``
    - ``maskInC``     ``[*L, 1, ny, nx]`` or None (OBCS Stevens interior; 1 if None)

    Returns the tracer tendency ``[*L, nz, ny, nx]`` (units tracer/s).
    """
    # Zonal diffusive flux at U-faces 0..nx (drF-stripped: dyG, not xA=dyG*drF).
    # DFx(i) uses T(i) - T(i-1) = tracer_x[..., i+1] - tracer_x[..., i].
    dfx = (
        -diffKh
        * dyG_x
        * hFacW_x
        * recip_dxC_x
        * (tracer_x[..., 1 : nx + 2] - tracer_x[..., 0 : nx + 1])
    )
    # Meridional diffusive flux at V-faces 0..ny.
    dfy = (
        -diffKh
        * dxG_y
        * hFacS_y
        * recip_dyC_y
        * (tracer_y[..., 1 : ny + 2, :] - tracer_y[..., 0 : ny + 1, :])
    )
    # Flux divergence at centres 0..nx-1 / 0..ny-1.
    div = (dfx[..., 1 : nx + 1] - dfx[..., 0:nx]) + (
        dfy[..., 1 : ny + 1, :] - dfy[..., 0:ny, :]
    )
    if maskInC is not None:
        div = div * maskInC
    return -recip_rA * recip_hFacC * div


def _tracer_del2_flux_form(
    tracer_x: torch.Tensor,
    tracer_y: torch.Tensor,
    *,
    dyG_x: torch.Tensor,
    dxG_y: torch.Tensor,
    recip_dxC_x: torch.Tensor,
    recip_dyC_y: torch.Tensor,
    hFacW_x: torch.Tensor,
    hFacS_y: torch.Tensor,
    recip_rA: torch.Tensor,
    recip_hFacC: torch.Tensor,
    maskInW: torch.Tensor | None,
    maskInS: torch.Tensor | None,
    nx: int,
    ny: int,
) -> torch.Tensor:
    """MITgcm GAD_GRAD_X/Y and GAD_DEL2: the centre ``del2`` of the biharmonic term.

    ``gad_grad_x.F:56-69`` / ``gad_grad_y.F:56-73`` and ``gad_del2.F:45-61`` with
    ``drF`` cancelled (``xA`` carries it, ``del2`` divides by it)::

        G_x(i)  = dyG hFacW recip_dxC (T(i) - T(i-1)) maskInW        U faces 0..nx
        del2    = recip_rA recip_hFacC ((G_x(i+1) - G_x(i)) + (G_y(j+1) - G_y(j)))

    ``maskInW``/``maskInS`` are ``None`` on a grid without them (MITgcm's factor
    is 1); ``del2`` carries no mask.  Inputs and shapes as
    :func:`horizontal_tracer_diffusion_flux_form`.
    """
    gx = dyG_x * hFacW_x * recip_dxC_x * (tracer_x[..., 1 : nx + 2] - tracer_x[..., 0 : nx + 1])
    if maskInW is not None:
        gx = gx * maskInW
    gy = (dxG_y * hFacS_y * recip_dyC_y
          * (tracer_y[..., 1 : ny + 2, :] - tracer_y[..., 0 : ny + 1, :]))
    if maskInS is not None:
        gy = gy * maskInS
    return recip_rA * recip_hFacC * (
        (gx[..., 1 : nx + 1] - gx[..., 0:nx]) + (gy[..., 1 : ny + 1, :] - gy[..., 0:ny, :])
    )


def horizontal_tracer_biharmonic_flux_form(
    del2_x: torch.Tensor,
    del2_y: torch.Tensor,
    *,
    dyG_x: torch.Tensor,
    dxG_y: torch.Tensor,
    recip_dxC_x: torch.Tensor,
    recip_dyC_y: torch.Tensor,
    hFacW_x: torch.Tensor,
    hFacS_y: torch.Tensor,
    recip_rA: torch.Tensor,
    recip_hFacC: torch.Tensor,
    diffK4: torch.Tensor,
    nx: int,
    ny: int,
) -> torch.Tensor:
    """MITgcm GAD_BIHARM_X/Y and the gad_calc_rhs divergence: ``-diffK4 del4 T``.

    ``gad_biharm_x.F:48-61`` / ``gad_biharm_y.F:51-65`` and
    ``gad_calc_rhs.F:770-777`` with ``drF`` cancelled::

        F4_x(i) = diffK4 dyG hFacW recip_dxC (del2(i) - del2(i-1))   U faces 0..nx
        gT      = -recip_rA recip_hFacC ((F4_x(i+1) - F4_x(i)) + (F4_y(j+1) - F4_y(j)))

    ``del2_x``/``del2_y`` are :func:`_tracer_del2_flux_form` haloed one cell
    along x and y by the caller; the fluxes carry no mask.
    """
    dfx = diffK4 * dyG_x * hFacW_x * recip_dxC_x * (del2_x[..., 1 : nx + 2] - del2_x[..., 0 : nx + 1])
    dfy = (diffK4 * dxG_y * hFacS_y * recip_dyC_y
           * (del2_y[..., 1 : ny + 2, :] - del2_y[..., 0 : ny + 1, :]))
    div = (dfx[..., 1 : nx + 1] - dfx[..., 0:nx]) + (dfy[..., 1 : ny + 1, :] - dfy[..., 0:ny, :])
    return -recip_rA * recip_hFacC * div


@dataclass(frozen=True)
@register_dynamics_module("horizontal_tracer_diffusion")
class HorizontalTracerDiffusion(TendencyModule):
    """Laplacian horizontal diffusion for tracers: κ_h ∇² T.

    Uses MITgcm-style hFac-aware flux-divergence stencil (GAD_DIFF_X/Y):
      DFx = -diffKh * hFacW * dyG * drF * (T_i - T_{i-1}) / dxC
      tendency = -(1 / hFacC) * div(DFx, DFy) / (drF * rA)

    On a uniform grid drF, rA, dyG, dxC all cancel between flux and
    divergence, leaving the hFac factors as the only difference from a
    plain Laplacian.  When no partial cells exist the result is identical
    to the previous ``laplacian_center`` implementation.
    """

    diffusivity: float = 1.0e-2
    # MITgcm diffK4T = diffK4S (m^4/s): the biharmonic term, one coefficient for
    # every tracer like ``diffusivity``.  0.0 => no biharmonic operation at all.
    diffusivity_k4: float = 0.0
    tracer_names: tuple[str, ...] = ("temp",)
    name: str = "horizontal_tracer_diffusion"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW
    tendency_phase: TendencyPhase = TendencyPhase.TRACER

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        temp_tend = torch.zeros_like(state.temp.data)
        salt_tend = torch.zeros_like(state.salt.data)

        if "temp" in self.tracer_names:
            temp_tend = self._diffuse(state.temp.data, grid)
        if "salt" in self.tracer_names:
            salt_tend = self._diffuse(state.salt.data, grid)

        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=torch.zeros_like(state.u.data),
            v=torch.zeros_like(state.v.data),
            temp=temp_tend,
            salt=salt_tend,
        )

    def _diffuse(self, tracer: torch.Tensor, grid: StructuredCGrid) -> torch.Tensor:
        """κ_h ∇²T, plus -κ_4 ∇⁴T when ``diffusivity_k4`` is set."""
        tendency = self._diffuse_harmonic(tracer, grid)
        if self.diffusivity_k4 != 0.0:
            tendency = tendency + self._diffuse_biharmonic(tracer, grid)
        return tendency

    def _diffuse_biharmonic(self, tracer: torch.Tensor, grid: StructuredCGrid) -> torch.Tensor:
        """-κ_4 ∇⁴T in MITgcm's full flux form (GAD_GRAD, GAD_DEL2, GAD_BIHARM) on
        every grid this module takes; ``SPEC_diffk4`` of the operator refactor.

        Exchanges, in order: the tracer (U, one i-halo column each side, then V,
        one j-halo row), the face fractions ``hFacW``/``hFacS`` the same way, then
        ``del2`` (U along x, V along y), all replicated at an uncoupled edge.
        MITgcm adds the second- and fourth-order fluxes before one divergence;
        the two divergences here are summed afterwards (summation order only).
        """
        # ponytail: the tracer and hFac halos repeat the harmonic pass's exchanges;
        # share them with _diffuse_harmonic if the extra exchanges show in a profile.
        nx = grid.nx
        ny = grid.ny
        dtype, device = tracer.dtype, tracer.device
        if grid.masks.has_3d_masks and grid.masks.hFacC is not None:
            hFacC = grid.masks.hFacC.to(dtype=dtype, device=device)
            hFacW = grid.masks.hFacW.to(dtype=dtype, device=device)
            hFacS = grid.masks.hFacS.to(dtype=dtype, device=device)
            if hFacC.ndim == 3:
                hFacC = hFacC.unsqueeze(0)
                hFacW = hFacW.unsqueeze(0)
                hFacS = hFacS.unsqueeze(0)
        else:
            nz = tracer.shape[-3]
            hFacC = torch.ones(1, nz, ny, nx, dtype=dtype, device=device)
            hFacW = torch.ones(1, nz, ny, nx + 1, dtype=dtype, device=device)
            hFacS = torch.ones(1, nz, ny + 1, nx, dtype=dtype, device=device)
        x_halo = HaloExtentV1(west_left_extent=1, east_right_extent=1,
                              south_bottom_extent=0, north_top_extent=0)
        y_halo = HaloExtentV1(west_left_extent=0, east_right_extent=0,
                              south_bottom_extent=1, north_top_extent=1)
        tracer_x = exchange(tracer, grid=grid, location="U", extent=x_halo,
                            fill=FillSpec.REPLICATE)
        tracer_y = exchange(tracer, grid=grid, location="V", extent=y_halo,
                            fill=FillSpec.REPLICATE)
        hFacW_x = exchange(hFacW[..., :nx], grid=grid, location="U", extent=x_halo,
                           fill=FillSpec.REPLICATE, constant=not hFacW.requires_grad)
        hFacS_y = exchange(hFacS[..., :ny, :], grid=grid, location="V", extent=y_halo,
                           fill=FillSpec.REPLICATE, constant=not hFacS.requires_grad)
        dxC = grid.metrics.dxC_for(dtype, device).view(1, 1, ny, nx + 1)
        dyC = grid.metrics.dyC_for(dtype, device).view(1, 1, ny + 1, nx)
        rA = grid.metrics.rA_for(dtype, device).view(1, 1, ny, nx)
        metrics = dict(
            dyG_x=grid.metrics.dyG_for(dtype, device).view(1, 1, ny, nx + 1),
            dxG_y=grid.metrics.dxG_for(dtype, device).view(1, 1, ny + 1, nx),
            recip_dxC_x=torch.where(dxC != 0, 1.0 / dxC, torch.zeros_like(dxC)),
            recip_dyC_y=torch.where(dyC != 0, 1.0 / dyC, torch.zeros_like(dyC)),
            hFacW_x=hFacW_x[..., 1 : nx + 2],
            hFacS_y=hFacS_y[..., 1 : ny + 2, :],
            recip_rA=torch.where(rA != 0, 1.0 / rA, torch.zeros_like(rA)),
            recip_hFacC=_positive_reciprocal(hFacC),
            nx=nx,
            ny=ny,
        )
        maskInW = grid.masks.maskInW
        maskInS = grid.masks.maskInS
        del2 = _tracer_del2_flux_form(
            tracer_x, tracer_y,
            maskInW=None if maskInW is None else maskInW.to(dtype=dtype, device=device),
            maskInS=None if maskInS is None else maskInS.to(dtype=dtype, device=device),
            **metrics,
        )
        del2_x = exchange(del2, grid=grid, location="U", extent=x_halo,
                          fill=FillSpec.REPLICATE)
        del2_y = exchange(del2, grid=grid, location="V", extent=y_halo,
                          fill=FillSpec.REPLICATE)
        return horizontal_tracer_biharmonic_flux_form(
            del2_x, del2_y, diffK4=tracer.new_tensor(self.diffusivity_k4), **metrics,
        )

    def _diffuse_harmonic(self, tracer: torch.Tensor, grid: StructuredCGrid) -> torch.Tensor:
        """Compute κ_h ∇²T with hFac-aware stencil."""
        nx = grid.nx
        ny = grid.ny
        diffusivity = tracer.new_tensor(self.diffusivity)

        if grid.masks.has_3d_masks and grid.masks.hFacC is not None:
            hFacC = grid.masks.hFacC.to(dtype=tracer.dtype, device=tracer.device)
            hFacW = grid.masks.hFacW.to(dtype=tracer.dtype, device=tracer.device)
            hFacS = grid.masks.hFacS.to(dtype=tracer.dtype, device=tracer.device)
            if hFacC.ndim == 3:
                hFacC = hFacC.unsqueeze(0)
                hFacW = hFacW.unsqueeze(0)
                hFacS = hFacS.unsqueeze(0)
        else:
            if (
                grid.metrics.is_uniform_x
                and grid.metrics.is_uniform_y
                and grid.metrics.has_uniform_row_geometry
            ):
                # Row-uniform Cartesian only: spherical grids must take the
                # full flux form below.
                return diffusivity * laplacian_center(tracer, grid)
            nz = tracer.shape[-3]
            hFacC = torch.ones(1, nz, ny, nx, dtype=tracer.dtype, device=tracer.device)
            hFacW = torch.ones(1, nz, ny, nx + 1, dtype=tracer.dtype, device=tracer.device)
            hFacS = torch.ones(1, nz, ny + 1, nx, dtype=tracer.dtype, device=tracer.device)

        x_halo = HaloExtentV1(
            west_left_extent=1,
            east_right_extent=1,
            south_bottom_extent=0,
            north_top_extent=0,
        )
        y_halo = HaloExtentV1(
            west_left_extent=0,
            east_right_extent=0,
            south_bottom_extent=1,
            north_top_extent=1,
        )
        tracer_x = exchange(
            tracer,
            grid=grid,
            location="U",
            extent=x_halo,
            fill=FillSpec.REPLICATE,
        )
        tracer_y = exchange(
            tracer,
            grid=grid,
            location="V",
            extent=y_halo,
            fill=FillSpec.REPLICATE,
        )
        # Exchange the non-duplicated face metrics so periodic east/north
        # edge faces retain the old west/south mirror metric.
        # hFac is a constant unless the bathymetry is a control (then it
        # requires grad on every tile, so the flag is rank-uniform).
        hFacW_x = exchange(
            hFacW[..., :nx],
            grid=grid,
            location="U",
            extent=x_halo,
            fill=FillSpec.REPLICATE,
            constant=not hFacW.requires_grad,
        )
        hFacS_y = exchange(
            hFacS[..., :ny, :],
            grid=grid,
            location="V",
            extent=y_halo,
            fill=FillSpec.REPLICATE,
            constant=not hFacS.requires_grad,
        )

        if (
            not grid.metrics.is_uniform_x
            or not grid.metrics.is_uniform_y
            or not grid.metrics.has_uniform_row_geometry
        ):
            # Full GAD_DIFF flux form for any j-varying geometry (spherical
            # rows included): dyG/dxC, dxG/dyC,
            # recip_rA per MIT gad_diff_x/y + gad_calc_rhs.
            dxC = grid.metrics.dxC_for(tracer.dtype, tracer.device).view(1, 1, ny, nx + 1)
            dyC = grid.metrics.dyC_for(tracer.dtype, tracer.device).view(1, 1, ny + 1, nx)
            dyG = grid.metrics.dyG_for(tracer.dtype, tracer.device).view(1, 1, ny, nx + 1)
            dxG = grid.metrics.dxG_for(tracer.dtype, tracer.device).view(1, 1, ny + 1, nx)
            rA = grid.metrics.rA_for(tracer.dtype, tracer.device).view(1, 1, ny, nx)
            recip_hFacC = _positive_reciprocal(hFacC)
            return horizontal_tracer_diffusion_flux_form(
                tracer_x,
                tracer_y,
                dyG_x=dyG,
                dxG_y=dxG,
                recip_dxC_x=torch.where(dxC != 0, 1.0 / dxC, torch.zeros_like(dxC)),
                recip_dyC_y=torch.where(dyC != 0, 1.0 / dyC, torch.zeros_like(dyC)),
                hFacW_x=hFacW_x[..., 1 : nx + 2],
                hFacS_y=hFacS_y[..., 1 : ny + 2, :],
                recip_rA=torch.where(rA != 0, 1.0 / rA, torch.zeros_like(rA)),
                recip_hFacC=recip_hFacC,
                diffKh=diffusivity,
                nx=nx,
                ny=ny,
            )

        dxC_row_1d = grid.metrics.dxC_row_for(tracer.dtype, tracer.device)
        _lead = max(tracer.ndim - 2, 0)
        dx = dxC_row_1d.view(*([1] * _lead), ny, 1)
        dy = tracer.new_tensor(grid.metrics.dyG_scalar_1d)

        # 🔴 OPEN PARTIAL-CELL DISAGREEMENT — read before changing either branch.
        # Both routes DO apply a final 1/hFacC today: this one through
        # ``_horizontal_tracer_diffusion_tendency_from_haloes`` (:1618-1619) and
        # the flux form above through ``recip_hFacC`` (:1703).  They agree with
        # each other to FP floor on shared geometry.
        #
        # A prior step-one comparison reported that
        # the form WITHOUT 1/hFacC matched MIT bitwise at partial cells, and
        # that adding it cost -6.78e-3 K at k=42, j=205, i=373 (hFacC=0.33); its
        # reading was that gad_calc_rhs.F's diffusion contribution to gT_loc
        # cancels against hFacW in the flux, leaving no net 1/hFacC.  MIT's
        # source visibly carries _recip_hFacC in that RHS block
        # (references/MITgcm/pkg/generic_advdiff/gad_calc_rhs.F:764-779), so the
        # source read and the measurement point opposite ways and neither has
        # been settled.  The helper this guards does apply 1/hFacC.
        #
        # Resolving the source-versus-measurement disagreement
        # needs an isolated MIT GAD_DIFF partial-cell dump, not another reading
        # of either side; until then do not flip one branch on its own.
        return _horizontal_tracer_diffusion_tendency_from_haloes(
            tracer,
            tracer_x,
            tracer_y,
            hFacC,
            hFacW_x,
            hFacS_y,
            diffusivity,
            dx,
            dy,
        )
