"""Horizontal momentum viscosity: the HorizontalViscosity tendency module (adapter).

This module reads the grid, exchanges halos stage by stage, selects the scheme
and calls the kernels of ``viscosity_geometry`` (wet fractions, corner hFacZ,
lengths), ``momentum_invariants`` (divergence, vorticity, strain, tension and
their gradient measures), ``viscosity_coefficients`` (Leith, Smagorinsky, base,
grid limits), ``divvort_viscosity`` (mom_vi_hdissip stencil),
``flux_viscosity`` (flux-form Laplacian) and ``lateral_drag`` (no-slip side
drag).  Tendencies are ``u`` ``[B, nz, ny, nx+1]`` and ``v`` ``[B, nz, ny+1, nx]``
in m/s^2; ``viscAh_W`` is published as a diagnostic for the variable schemes.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from locomgcm.geometry import StructuredCGrid
from locomgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange
from locomgcm.state import DiagnosticState, PrognosticState, StateTendency

from .base import MomentumTermClass, TendencyModule, TendencyStepDomain
from ._horizontal_diffusion_topology import (
    _any_edge_couples,
    _edge_couples,
    _has_distributed_neighbor,
    _divvort_curl_halos,
    _has_internal_distributed_neighbor,
    divvort_tendency_for_grid,
)
from .divvort_viscosity import (
    _apply_del2uv_interior_masks,
    _divvort_corner_flux,
    _divvort_variable_fluxes,
    _u_divvort_all_from_haloed,
    _u_divvort_interior_from_fluxes,
    _u_divvort_output_from_all,
    _u_divvort_output_from_interior,
    _v_divvort_all_from_haloed,
    _v_divvort_interior_from_fluxes,
    _v_divvort_output_from_all,
    _v_divvort_output_from_interior,
)
from .flux_viscosity import (
    _u_center_strain_flux,
    _u_corner_strain_divergence_metric,
    _u_corner_strain_flux,
    _u_corner_strain_flux_with_edges,
    _u_del2,
    _u_del2_side_term,
    _u_del2_zonal_flux,
    _u_laplacian_row_divergence,
    _u_strain_flux_metric,
    _u_viscous_divergence,
    _u_viscous_meridional_divergence,
    _u_viscous_meridional_divergence_biharmonic,
    _u_viscous_zonal_flux,
    _u_viscous_zonal_flux_biharmonic,
    _v_center_strain_flux,
    _v_corner_strain_divergence_metric,
    _v_corner_strain_flux,
    _v_corner_strain_flux_with_edges,
    _v_del2,
    _v_del2_meridional_flux,
    _v_del2_side_term,
    _v_laplacian_row_divergence,
    _v_strain_flux_metric,
    _v_viscous_divergence,
    _v_viscous_meridional_flux,
    _v_viscous_meridional_flux_biharmonic,
    _v_viscous_zonal_divergence,
    _v_viscous_zonal_divergence_biharmonic,
    _x_face_difference,
    _x_flux_divergence,
    _y_face_difference,
    _y_flux_divergence,
)
from .lateral_drag import (
    _side_drag_u_2d,
    _side_drag_v_2d,
    _variable_side_drag_tendencies,
)
from .momentum_invariants import (
    DivVortInvariants,
    _apply_hfacz_side_mask_to_corner_fields,
    _d_point_divergence_gradient_measures,
    _d_point_divergence_gradient_measures_from_faces,
    _d_point_smag_strain_magnitude,
    _d_point_vorticity_gradient_measures,
    _hdiv_from_face_velocities,
    _hdiv_gradient_faces_from_haloes,
    _hdiv_gradients_from_center,
    _leith_vorticity_gradients_from_masks,
    _tension_from_face_velocities,
    _vort3_strain_from_haloes,
    _z_point_divergence_gradient_measures,
    _z_point_divergence_gradient_measures_from_haloes,
    _z_point_smag_strain_magnitude,
    _z_point_smag_strain_magnitude_from_halo,
    _z_point_vorticity_gradient_measures,
    _z_point_vorticity_gradient_measures_from_haloes,
)
from .viscosity_coefficients import (
    ViscosityMeasures,
    _add_biharmonic_base_viscosity,
    _add_harmonic_base_viscosity,
    _biharmonic_grid_rdt_scale,
    _biharmonic_leith_full,
    _biharmonic_leith_maxnorm,
    _biharmonic_length_powers,
    _biharmonic_viscosity_with_smag,
    _grid_cap,
    _grid_floor,
    _harmonic_grid_rdt_scale,
    _harmonic_leith_full,
    _harmonic_leith_maxnorm,
    _harmonic_leith_vorticity,
    _harmonic_viscosity_with_smag,
    _require_div_measures,
    _w_viscosity_from_center_viscosity,
)
from .viscosity_geometry import (
    DivVortGeometry,
    _compute_hFacZ_4neighbor,
    _constant_cache_enabled,
    _divvort_corner_hfacz,
    _grid_clamp_length_scales,
    _leith_length_scales,
    _recip,
    _side_drag_closed_fractions,
    _strain_metric_flags,
    divvort_geometry,
)
from .strain_tension_viscosity import (
    _strain_tension_center_flux,
    _strain_tension_corner_flux,
    _u_strain_tension_dissipation,
    _v_strain_tension_dissipation,
)
from locomgcm.dynamics._registry import register_dynamics_module


#: Schemes whose harmonic viscosity is MOM_CALC_VISC's variable viscAh_D / viscAh_Z
#: (the base viscosity, Leith, Smagorinsky and the grid limits): ``leith_smag`` applies it
#: in the div-vort form of the vector-invariant momentum (mom_vi_hdissip.F),
#: ``leith_smag_flux`` in the flux-divergence Laplacian of the flux-form momentum
#: (mom_fluxform.F:454, 612-618), ``leith_smag_strain_tension`` in the strain-tension
#: form of the vector-invariant momentum (useStrainTensionVisc, mom_hdissip.F).
VARIABLE_VISCOSITY_SCHEMES = ("leith_smag", "leith_smag_flux", "leith_smag_strain_tension")

#: Every ``viscosity_scheme`` HorizontalViscosity accepts, one entry per branch of
#: :meth:`HorizontalViscosity.evaluate`.
VISCOSITY_SCHEMES = (
    "constant", "divvort", "leith_smag", "leith_smag_flux", "leith_smag_strain_tension",
)

#: Fields only MOM_CALC_VISC reads (the variable coefficients and grid limits of
#: ``leith_smag`` and ``leith_smag_flux``); any other scheme refuses them nonzero.
VARIABLE_VISCOSITY_FIELDS = (
    "visc_c2_leith", "visc_c2_leith_d", "visc_c2_smag",
    "visc_ah_grid_min", "visc_ah_grid_max", "use_full_leith",
    "visc_c4_leith", "visc_c4_leith_d", "visc_c4_smag",
    "visc_a4_grid_min", "visc_a4_grid_max",
)


@dataclass(frozen=True)
@register_dynamics_module("horizontal_viscosity")
class HorizontalViscosity(TendencyModule):
    """Laplacian horizontal viscosity using direct face-based stencil.

    Matches MITgcm's MOM_U_DEL2U / MOM_V_DEL2V discretisation:
      del2u = (1/hFacW) * div[ hFac-weighted strain fluxes ]

    Strain fluxes:
      x-direction: at cell centers, weighted by hFacC
      y-direction: at corners (vorticity points), weighted by hFacZ

    When ``no_slip_sides`` is True (MITgcm default ``no_slip_sides=.TRUE.``),
    an additional side-drag term is applied at wall-adjacent faces via
    ``MOM_U_SIDEDRAG`` / ``MOM_V_SIDEDRAG``, using a ghost-cell image
    that doubles the velocity gradient.  ``side_drag_factor`` controls
    the multiplier (MITgcm default: 2.0).
    """

    viscosity: float = 1.0e-2
    no_slip_sides: bool = True
    side_drag_factor: float = 2.0
    # "divvort": the constant viscosity in mom_vi_hdissip.F's div-vort form;
    # "leith_smag": MOM_CALC_VISC's variable viscosity in that div-vort form (MITgcm's
    # vector-invariant momentum); "leith_smag_flux": the same viscosity in the flux-form
    # Laplacian (MITgcm's flux-form momentum, mom_fluxform.F:612-618);
    # "leith_smag_strain_tension": the same viscosity in the strain-tension form
    # (MITgcm's vector-invariant momentum with useStrainTensionVisc, mom_hdissip.F);
    # "constant": the constant viscosity in the flux-form Laplacian (MOM_U_DEL2U-style
    # stencil).  Any other value is refused (VISCOSITY_SCHEMES).
    viscosity_scheme: str = "constant"
    visc_c2_leith: float = 0.0
    visc_c2_smag: float = 0.0
    # MITgcm viscAhGridMin/Max (mom_calc_visc.F:514-517): grid+dt-scaled
    # floor/cap on the harmonic viscosity.  Per cell:
    #   viscAh = clamp(base+Leith,
    #                  viscAhGridMin * 0.25 * L^2/dt,
    #                  viscAhGridMax * 0.25 * L^2/dt)
    # with L^2 = 2/(dxF^-2 + dyF^-2) at D (centre) points and
    # 2/(dxV^-2 + dyU^-2) at Z (corner) points (mom_init_fixed.F:96-129,
    # useAreaViscLength=.FALSE.; equal to rA/rAz only where dx = dy).  This is
    # Gong's grid-scale-dissipation floor: at L=500m, dt=10s, viscAhGridMin=
    # 0.00032 gives a 2.0 m^2/s floor (vs a bare base of 0.01) that squelches
    # the 2dx computational mode (Bryan 1975, Re_h<2).  0.0 = disabled
    # (back-compat: existing cases are byte-identical).  Requires dt>0 when set.
    visc_ah_grid_min: float = 0.0
    visc_ah_grid_max: float = 0.0
    # ── Grid-scale dissipation completion (MITgcm mom_calc_visc.F), all
    # default-off so existing leith_smag cases stay byte-identical ──
    # full-Leith (MIT useFullLeith): the L2-norm sqrt branch that combines the
    # vorticity AND divergence gradient magnitudes, with the ^6 coefficient
    # powers, instead of the default ^3 max-norm vorticity-only branch.
    use_full_leith: bool = False
    # divergence Leith (MIT viscC2LeithD): the leithD2fac*grad(hDiv) term that
    # damps grid-scale divergence (w) instabilities; added in BOTH the max-norm
    # and the full-Leith branches.  0.0 => term vanishes (byte-identical).
    visc_c2_leith_d: float = 0.0
    # Biharmonic viscA4 (MIT viscA4D/Z + viscC4Leith/LeithD/Smag): del^4 = del^2
    # of del^2.  'constant' (flux form) and 'divvort' (div-vort form) apply
    # viscosity_a4 as a constant; 'leith_smag'/'leith_smag_flux' add visc_c4_* and the
    # grid limits of MOM_CALC_VISC (VARIABLE_VISCOSITY_FIELDS).  0.0 everywhere =>
    # operator skipped.  The cubed sphere refuses it (_refuse_cube_biharmonic).
    viscosity_a4: float = 0.0
    visc_c4_leith: float = 0.0
    visc_c4_leith_d: float = 0.0
    visc_c4_smag: float = 0.0
    visc_a4_grid_min: float = 0.0
    visc_a4_grid_max: float = 0.0
    dt: float = 0.0
    name: str = "horizontal_viscosity"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW
    momentum_term_class = MomentumTermClass.DISSIPATION
    # Topology capability (locomgcm/grid_support.py).
    # Cube claim backed by a bitwise registry-injection check (constant-Ah
    # divvort path); the variable-viscosity schemes are refused on the cube
    # (_refuse_cube_variable_viscosity).
    grid_support = ("latlon", "cube")

    def __post_init__(self) -> None:
        if self.viscosity_scheme not in VISCOSITY_SCHEMES:
            raise ValueError(
                f"horizontal_viscosity: viscosity_scheme={self.viscosity_scheme!r} is not "
                f"one of {VISCOSITY_SCHEMES}"
            )
        if self.viscosity_scheme not in VARIABLE_VISCOSITY_SCHEMES:
            unread = [f for f in VARIABLE_VISCOSITY_FIELDS if getattr(self, f) != 0.0]
            if unread:
                raise ValueError(
                    f"horizontal_viscosity: viscosity_scheme={self.viscosity_scheme!r} does "
                    f"not read {unread}; MITgcm applies them in MOM_CALC_VISC only "
                    f"(mom_fluxform.F:453-458, mom_vecinv.F:373-381), i.e. under "
                    f"{VARIABLE_VISCOSITY_SCHEMES}"
                )
        if self.viscosity_scheme == "leith_smag_strain_tension" and self._biharmonic_active():
            raise ValueError(
                "horizontal_viscosity: viscosity_scheme='leith_smag_strain_tension' has no "
                "biharmonic term; MITgcm stops on it (mom_hdissip.F:84-86), so viscosity_a4 "
                "and visc_c4_* must be 0"
            )

    def _refuse_cube_biharmonic(self, grid) -> None:
        """The cubed-sphere path is the harmonic div-vort stencil only.

        ``divvort_tendency_for_grid`` returns that tendency for ``face_count == 6``
        before the scheme branches, so a biharmonic coefficient would be dropped on
        U/V.  The W counterpart, :meth:`visca4_w_for_nonhydrostatic`, builds
        ``viscA4_W`` on any grid, so the refusal sits on the U/V tendency.
        """
        # ponytail: the cube biharmonic needs del2u/del2v and their hDiv/vort3 across
        # the face seams (FILL_CS_CORNER_TR_RL in mom_vi_del2uv.F:81, :106); add it with
        # a cube case that uses it.
        if self._biharmonic_active() and getattr(grid, "face_count", 1) == 6:
            raise NotImplementedError(
                "horizontal_viscosity: the cubed-sphere path has no U/V biharmonic term; "
                "set viscosity_a4 to 0"
            )

    def _refuse_cube_variable_viscosity(self, grid) -> None:
        """The cubed-sphere path is the constant-viscosity div-vort stencil only.

        ``divvort_tendency_for_grid`` returns that tendency for ``face_count == 6``
        before the scheme branches, so a Leith or Smagorinsky coefficient would be
        dropped without a word.  MITgcm computes the variable viscosities on every
        tile, cubed sphere included (pkg/mom_common/mom_calc_visc.F:483-517, no
        topology condition); LOCOMgcm has no cube implementation of them, so the
        pairing is refused rather than run with the constant viscosity.
        """
        # ponytail: the cube variable viscosity needs the coefficient fields on the
        # cube halo (exchange of viscAh_D/Z across face seams) and the corner rules of
        # the cube div-vort stencil; add it with a cube case that uses it.
        if (getattr(grid, "face_count", 1) == 6
                and self.viscosity_scheme in VARIABLE_VISCOSITY_SCHEMES):
            raise NotImplementedError(
                f"horizontal_viscosity: viscosity_scheme={self.viscosity_scheme!r} on the "
                "cubed sphere: the cube path applies the constant viscosity only and would "
                "drop the Leith and Smagorinsky coefficients that MITgcm computes there "
                "(pkg/mom_common/mom_calc_visc.F:483-517); use 'divvort'"
            )

    def prepare_compiled_resume_constants(self, grid, *, dtype, device) -> None:
        """Restore the cached Leith geometry before tracing a continued run."""
        if (self.viscosity_scheme in VARIABLE_VISCOSITY_SCHEMES
                and grid.face_count == 1 and _constant_cache_enabled()):
            _leith_length_scales(
                grid,
                dxF=grid.metrics.dxF_for(dtype, device),
                dyF=grid.metrics.dyF_for(dtype, device),
                dxV=grid.metrics.dxV_for(dtype, device),
                dyU=grid.metrics.dyU_for(dtype, device),
                dtype=dtype,
                device=device,
                ny=grid.ny,
                nx=grid.nx,
            )

    def prepare_host_constants(self, grid, *, dtype, device) -> None:
        """Resolve strain-stencil branches before the compiled step."""
        self._refuse_cube_variable_viscosity(grid)
        if (getattr(grid, "face_count", 1) == 1
                and self.viscosity_scheme not in ("divvort", *VARIABLE_VISCOSITY_SCHEMES)):
            _strain_metric_flags(grid, dtype=dtype, device=device)

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        u = state.u.data  # [B, nz, ny, nx+1]
        v = state.v.data  # [B, nz, ny+1, nx]

        viscosity = u.new_tensor(self.viscosity)

        self._refuse_cube_biharmonic(grid)
        self._refuse_cube_variable_viscosity(grid)
        topology_tendency = divvort_tendency_for_grid(
            state,
            grid,
            viscosity,
            no_slip_sides=self.no_slip_sides,
            side_drag_factor=self.side_drag_factor,
        )
        if topology_tendency is not None:
            return topology_tendency

        if self.viscosity_scheme == "leith_smag":
            # ---- Fused Leith/Smag accelerator boundary (hvisc_fused_cuda) ----
            # Two CUDA launches replace these stages of
            # the eager block below (hDiv, the corner curl pair, the dry-corner
            # side mask, tension, the Leith vorticity gradients and both point
            # measures, both Smag strain magnitudes, the harmonic assembly with
            # its base viscosity, and the div-vort dissipation stencil).  The
            # dispatcher returns None whenever it cannot reproduce this
            # configuration bitwise, and the frozen eager block underneath is
            # then the single fallback implementation.  `viscAh_D` comes back
            # from the dispatcher because the viscAh_W / viscAh3d_00
            # diagnostics stay on this side of the boundary.
            fused = None
            from locomgcm.accel import (
                consumer_snapshot,
                in_accelerated_dispatch,
                requested,
            )

            if not in_accelerated_dispatch():
                with consumer_snapshot():
                    if requested(("hvisc_fused_cuda",)):
                        from ._hvisc_fused import dispatcher as _hvisc_dispatcher

                        fused = _hvisc_dispatcher.hvisc_fused_dispatch(
                            self, u, v, grid
                        )

            if fused is not None:
                u_result, v_result, viscAh_D = fused
                diagnostics.set_tensor(
                    "viscAh_W",
                    _w_viscosity_from_center_viscosity(viscAh_D),
                )
                diagnostics.set_tensor(
                    "viscAh3d_00", diagnostics.tensors["viscAh_W"]
                )
            else:
                # ONE `_divvort_terms(u, v, grid, ...)` serves the whole leith_smag
                # branch: the coefficient assembly, the dissipation stencil and the
                # biharmonic chain all consumed the same (u, v, grid) triple and
                # each rebuilt the dict from scratch (3 identical passes/step).
                #
                # Bitwise argument for feeding the `side_mask_fac`-built dict to the
                # stencil, which previously recomputed it with the 0.0 default:
                # `side_mask_fac` reaches exactly two keys, `vort3_bc`/`strain_bc`
                # (:meth:`_divvort_terms` -> `_apply_hfacz_side_mask_to_corner_fields`),
                # and those two are read only by :meth:`_leith_grad_measures`.
                # :meth:`_divvort_apply_variable` reads `hDiv`/`vort3`/`hFacZ` + metrics/masks,
                # none of which `side_mask_fac` touches — so the shared dict carries
                # the same bits the discarded recompute would have produced.
                side_mask_fac = self.side_drag_factor if self.no_slip_sides else 0.0
                terms = self._divvort_terms(u, v, grid, side_mask_fac=side_mask_fac)
                geom, hFacZ, inv = terms
                measures = self._leith_grad_measures(geom, inv, grid)
                viscAh_D, viscAh_Z = self._compute_leith_smag_viscosities_with_floor(
                    u, v, grid, terms=terms, measures=measures,
                )
                diagnostics.set_tensor(
                    "viscAh_W",
                    _w_viscosity_from_center_viscosity(viscAh_D),
                )
                diagnostics.set_tensor("viscAh3d_00", diagnostics.tensors["viscAh_W"])
                u_result, v_result = self._divvort_apply_variable(
                    geom, hFacZ, inv.hDiv, inv.vort3, viscAh_D, viscAh_Z, grid
                )
                if self.no_slip_sides:
                    u_side, v_side = self._variable_side_drag(
                        u, v, grid, geom, hFacZ, viscAh_Z
                    )
                    u_result = u_result + u_side
                    v_result = v_result + v_side
                if self._biharmonic_active():
                    u_bih, v_bih = self._biharmonic_dissipation(
                        u, v, grid, terms=terms, measures=measures,
                    )
                    u_result = u_result + u_bih
                    v_result = v_result + v_bih
        elif self.viscosity_scheme == "leith_smag_flux":
            u_result, v_result = self._flux_form_variable_dissipation(u, v, grid, diagnostics)
        elif self.viscosity_scheme == "leith_smag_strain_tension":
            u_result, v_result = self._strain_tension_variable_dissipation(
                u, v, grid, diagnostics,
            )
        elif self.viscosity_scheme == "divvort":
            if self._biharmonic_active():
                u_result, v_result = self._divvort_constant_biharmonic(
                    u, v, viscosity, grid,
                )
            else:
                u_result, v_result = self._divvort_dissipation(u, v, viscosity, grid)
                if self.no_slip_sides:
                    # _u/_v_side_drag return the FULL drag (viscosity + factor
                    # applied inside the unified 2-D-metric helper), so add directly.
                    u_result = u_result + self._u_side_drag(u, grid, self.viscosity)
                    v_result = v_result + self._v_side_drag(v, grid, self.viscosity)
        elif self.viscosity_scheme == "constant":
            if self._biharmonic_active():
                u_result, v_result = self._flux_form_constant_biharmonic(u, v, grid)
            else:
                u_tend = self._u_laplacian(u, grid)
                v_tend = self._v_laplacian(v, grid)
                u_result = viscosity * u_tend
                v_result = viscosity * v_tend
                if self.no_slip_sides:
                    u_result = u_result + self._u_side_drag(u, grid, self.viscosity)
                    v_result = v_result + self._v_side_drag(v, grid, self.viscosity)

        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=u_result,
            v=v_result,
            temp=torch.zeros_like(state.temp.data),
            salt=torch.zeros_like(state.salt.data),
        )

    def viscah_w_for_nonhydrostatic(
        self,
        state: PrognosticState,
        grid: StructuredCGrid,
    ) -> torch.Tensor:
        """Return MIT ``viscAh_W`` for CALC_GW W-momentum dissipation.

        For Leith/Smag, MITgcm computes variable center viscosity
        ``viscAh_D`` and hands it to ``viscAh_W`` on W levels via a top-level
        copy and vertical half-sums.  Constant-viscosity schemes use the scalar
        harmonic floor on every W cell.
        """
        if self.viscosity_scheme in VARIABLE_VISCOSITY_SCHEMES:
            viscAh_D, _ = self._compute_leith_smag_viscosities_with_floor(
                state.u.data,
                state.v.data,
                grid,
            )
            return _w_viscosity_from_center_viscosity(viscAh_D)
        return torch.full_like(state.temp.data, float(self.viscosity))

    def visca4_w_for_nonhydrostatic(
        self,
        state: PrognosticState,
        grid: StructuredCGrid,
    ) -> torch.Tensor | None:
        """Return MIT ``viscA4_W`` for the CALC_GW biharmonic W-momentum
        dissipation (``mom_calc_visc.F:691,712`` vertical half-sum of
        ``viscA4_D``).  ``None`` when biharmonic is inactive (del^4 of w skipped,
        existing NH cases byte-identical).  Assumes ``viscA4W=viscA4D`` (no
        separate W base ⇒ shiftA4=0), matching default configs."""
        if not self._biharmonic_active():
            return None
        if self.viscosity_scheme in VARIABLE_VISCOSITY_SCHEMES:
            viscA4_D, _ = self._compute_biharmonic_viscosities_with_floor(
                state.u.data, state.v.data, grid,
            )
        elif self.viscosity_scheme in ("constant", "divvort"):
            viscA4_D = torch.full_like(state.temp.data, float(self.viscosity_a4))
        return _w_viscosity_from_center_viscosity(viscA4_D)

    def _compute_leith_smag_viscosities_with_floor(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        grid: StructuredCGrid,
        terms: tuple[DivVortGeometry, torch.Tensor, DivVortInvariants] | None = None,
        measures: ViscosityMeasures | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        viscAh_D, viscAh_Z = self._compute_leith_smag_viscosities(
            u, v, grid, terms=terms, measures=measures,
        )
        # MITgcm `pkg/mom_common/mom_calc_visc.F:502,649`: variable
        # `viscAh_D/Z` includes the scalar harmonic base `viscAhD/Z`
        # (which inherits `viscAh` per `model/src/ini_parms.F:512-513`)
        # before adding Leith/Smag.  Without this floor LOCOMgcm under-dissipates
        # momentum wherever Leith+Smag is small, matching the d1b-only
        # ablation 326-366× degradation against MIT.
        base = u.new_tensor(self.viscosity)
        viscAh_D, viscAh_Z = _add_harmonic_base_viscosity(
            viscAh_D,
            viscAh_Z,
            base,
        )
        return self._apply_grid_visc_floor_cap(viscAh_D, viscAh_Z, grid)

    def _apply_grid_visc_floor_cap(
        self,
        viscAh_D: torch.Tensor,
        viscAh_Z: torch.Tensor,
        grid: "StructuredCGrid",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MITgcm viscAhGridMin/Max grid+dt-scaled floor/cap.

        ``mom_calc_visc.F:514-517`` (D points) / ``:661-664`` (Z points):
            L2rdt        = 0.25 * L2 / dt          (:385-386, :535-536)
            viscAh_DMin  = MAX(viscAhGridMin*L2rdt, Uscl)   # Uscl=0 (viscAhReMax=0)
            viscAh_D     = MAX(viscAh_DMin, Alin)           # Alin = base + Leith
            viscAh_DMax  = MIN(viscAhGridMax*L2rdt, viscAhMax)  # viscAhMax=1e21
            viscAh_D     = MIN(viscAh_DMax, viscAh_D)

        with ``L2 = L2_D`` at D and ``L2_Z`` at Z points, the
        ``useAreaViscLength=.FALSE.`` harmonic lengths of
        ``_grid_clamp_length_scales``; rA/rAz would be the
        ``useAreaViscLength=.TRUE.`` form.  i.e. ``viscAh = clamp(base+Leith,
        gridMin*0.25*L2/dt, gridMax*0.25*L2/dt)``.  No-op when both coefficients
        are 0 so existing cases stay byte-identical.
        """
        grid_min = float(self.visc_ah_grid_min)
        grid_max = float(self.visc_ah_grid_max)
        if grid_min <= 0.0 and grid_max <= 0.0:
            return viscAh_D, viscAh_Z
        if self.dt <= 0.0:
            raise ValueError(
                "horizontal_viscosity.visc_ah_grid_min/max require dt>0 "
                "(set `dt` in the horizontal_viscosity kwargs = stepper deltaT)"
            )
        l2_d, l2_z = _grid_clamp_length_scales(grid, viscAh_D.dtype, viscAh_D.device)
        l2rdt_d = _harmonic_grid_rdt_scale(l2_d, self.dt)             # [ny, nx]
        l2rdt_z = _harmonic_grid_rdt_scale(l2_z, self.dt)             # [ny+1, nx+1]
        if grid_min > 0.0:
            viscAh_D, viscAh_Z = _grid_floor(viscAh_D, viscAh_Z, l2rdt_d, l2rdt_z, grid_min)
        if grid_max > 0.0:
            viscAh_D, viscAh_Z = _grid_cap(viscAh_D, viscAh_Z, l2rdt_d, l2rdt_z, grid_max)
        return viscAh_D, viscAh_Z

    @staticmethod
    def _divvort_terms(
        u: torch.Tensor,
        v: torch.Tensor,
        grid: "StructuredCGrid",
        *,
        side_mask_fac: float = 0.0,
    ) -> tuple[DivVortGeometry, torch.Tensor, DivVortInvariants]:
        """Geometry, corner ``hFacZ`` and invariants of ``(u, v)``.

        Stages in order: read the metrics and wet fractions, form ``hDiv``,
        exchange the curl halos (``dxC``, ``u``, ``v*dyC``), build ``hFacZ``
        (its own memoized exchange), then the corner and centre invariants.
        """
        dtype = u.dtype
        device = u.device
        geom = divvort_geometry(grid, nz=u.shape[-3], dtype=dtype, device=device)
        nx = geom.nx
        ny = geom.ny

        # ── Step 1: hDiv (hDivScheme=2, mom_calc_hdiv.F:55-71) ──
        hDiv = _hdiv_from_face_velocities(
            u,
            v,
            dyG=geom.dyG.view(1, 1, ny, nx + 1),
            dxG=geom.dxG.view(1, 1, ny + 1, nx),
            hFacW=geom.hFacW,
            hFacS=geom.hFacS,
            recip_rA=geom.recip_rA,
            recip_hFacC=geom.recip_hFacC,
            center_mask=geom.maskInC,
        )

        # ── Step 2: vort3 (mom_calc_relvort3.F:50-61, no lopping) ──
        # vort3(i,j) = recip_rAz * [v(i)*dyC(i) - v(i-1)*dyC(i-1)
        #              - u(j)*dxC(j) + u(j-1)*dxC(j-1)]
        # At corner (i,j), shape [nz, ny+1, nx+1]
        dxC_ext, u_ext, v_dyC_ext = _divvort_curl_halos(
            u, v, grid, dxC=geom.dxC, dyC=geom.dyC,
        )
        vort3, strain = _vort3_strain_from_haloes(
            u_ext=u_ext,
            dxC_ext=dxC_ext,
            v_dyC_ext=v_dyC_ext,
            recip_rAz=geom.recip_rAz.view(1, 1, ny + 1, nx + 1),
            ny=ny,
            nx=nx,
        )

        # ── Step 3: hFacZ (hZoption=0, MIN of 4 neighbors) ──
        # Geometry constant — memoized per grid (Site 1, viscosity side).
        hFacZ = _divvort_corner_hfacz(grid, geom.hFacW, geom.hFacS, dtype, device)

        # MITgcm `vort3BC` / `strainBC` — boundary-conditioned vorticity and
        # strain consumed by variable-viscosity (Leith / Smag) construction.
        # Per `pkg/mom_vecinv/mom_vecinv.F:286-340`, MIT scales these fields
        # by `sideMaskFac` at dry `hFacZ==0` corners (0.0 for free-slip,
        # `sideDragFactor` for no-slip). Without this conditioning, raw
        # dry-corner vorticity contaminates the Leith gradients.
        vort3_bc, strain_bc = _apply_hfacz_side_mask_to_corner_fields(
            vort3,
            strain,
            hFacZ,
            side_mask_fac,
        )

        # MITgcm MOM_CALC_TENSION: tension is at C-points and uses metric
        # flux differences without hFac; masks are implicit in u/v contents.
        tension = _tension_from_face_velocities(
            u,
            v,
            dyF=geom.dyF.view(1, 1, ny, nx),
            dxG=geom.dxG.view(1, 1, ny + 1, nx),
            recip_rA=geom.recip_rA,
            center_mask=geom.maskInC,
        )

        return geom, hFacZ, DivVortInvariants(
            hDiv=hDiv,
            vort3=vort3,
            vort3_bc=vort3_bc,
            strain_bc=strain_bc,
            tension=tension,
        )

    def _leith_grad_measures(
        self,
        geom: DivVortGeometry,
        inv: DivVortInvariants,
        grid: "StructuredCGrid",
    ) -> ViscosityMeasures:
        """Vorticity & divergence gradient magnitudes + Smag strain + length
        scales at D (centre) and Z (corner) points — the shared inputs of the
        harmonic (viscAh) and biharmonic (viscA4) coefficient assembly.

        Mirrors MITgcm ``mom_calc_visc.F``: divDx/divDy (grad hDiv, no mask,
        lines 274-291), vrtDx/vrtDy (grad vort3 with maskS/W, 296-315), the
        max-norm grd*  (useFullLeith=.FALSE., 456-463 / 608-614) and the
        0.25*sum-of-squares grd*_sq (useFullLeith=.TRUE., 408-418 / 562-571),
        and the Smag strain magnitudes (483-491 / 633-641).  The D-vs-Z index
        offsets are the alignment crux.
        """
        nx = geom.nx
        ny = geom.ny
        dyG = geom.dyG
        dyC = geom.dyC
        dxC_row = geom.dxC
        dxG_ext = geom.dxG
        hDiv = inv.hDiv
        vort3 = inv.vort3_bc
        tension = inv.tension
        strain = inv.strain_bc
        maskW = geom.maskW
        maskS = geom.maskS
        # Seam semantics come from the grid topology (numerics rule 11): any
        # coupled edge, a periodic axis on one tile or a rank seam, takes the
        # exchange-based measures below.  The zero-padding measures treat every
        # outer neighbour as absent and are right only when all four edges are
        # walls or open boundaries; choosing them by "no distributed neighbour"
        # zeroed the periodic neighbours of a single-tile doubly periodic grid.
        seam_aware = _any_edge_couples(grid)
        # Gate: with divergence-Leith and biharmonic both off, nothing derived
        # from divDx/divDy is ever read.  Same predicate object as the consumer
        # branch in _harmonic_from_measures / _compute_biharmonic_viscosities.
        want_div = self._needs_leith_sq_and_div_measures()

        # ── divDx/divDy = grad(hDiv) at C-faces (no velocity mask) ──
        if not want_div:
            divDx = divDy = None
        elif seam_aware:
            # The neighbour centre across an edge comes from ``exchange``: the
            # wrap cell on a periodic axis of one tile, the neighbour rank's cell
            # across a rank seam.
            hDiv_ny, hDiv_nx = hDiv.shape[-2:]
            hDiv_x = exchange(
                hDiv,
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
            hDiv_y = exchange(
                hDiv,
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
            divDx, divDy = _hdiv_gradient_faces_from_haloes(
                hDiv_x,
                hDiv_y,
                dxC_row=dxC_row,
                dyG=dyC.view(1, 1, ny + 1, nx),
                ny=hDiv_ny,
                nx=hDiv_nx,
                west=_edge_couples(grid, "west"),
                east=_edge_couples(grid, "east"),
                south=_edge_couples(grid, "south"),
                north=_edge_couples(grid, "north"),
            )
        else:
            divDx, divDy = _hdiv_gradients_from_center(
                hDiv,
                dxC=dxC_row[:, 1:nx].view(1, 1, ny, nx - 1),
                dyG=dyC[1:ny, :].view(1, 1, ny - 1, nx),
            )

        # ── vrtDx/vrtDy = grad(vort3) at faces (maskS/maskInS, maskW/maskInW) ──
        vrtDx, vrtDy = self._leith_vorticity_gradients(
            vort3=vort3,
            dxG_ext=dxG_ext,
            dyG=dyG,
            maskS=maskS,
            maskW=maskW,
            grid=grid,
        )

        # ── D-point (centre) vorticity-gradient measures ──
        # MIT grdVrt_D over vrtDx(i,j),(i,j+1) and vrtDy(i,j),(i+1,j).
        vrt_d_max, grd_vrt_d_sq = _d_point_vorticity_gradient_measures(
            vrtDx,
            vrtDy,
            ny=ny,
            nx=nx,
            want_sq=want_div,
        )

        # ── D-point divergence-gradient measures ──
        # MIT grdDiv_D over divDx(i,j),(i+1,j) and divDy(i,j),(i,j+1).
        if not want_div:
            grd_div_d_max = grd_div_d_sq = None
        elif seam_aware:
            grd_div_d_max, grd_div_d_sq = _d_point_divergence_gradient_measures_from_faces(
                divDx,
                divDy,
            )
        else:
            grd_div_d_max, grd_div_d_sq = _d_point_divergence_gradient_measures(
                divDx,
                divDy,
            )

        # ── D-point Smag strain magnitude (tension primary, strain 4-avg) ──
        smag_d = _d_point_smag_strain_magnitude(tension, strain)

        # ── D- and Z-point length scales (useAreaViscLength=F harmonic mean) ──
        # Grid constant — memoized per grid (Site 3).  The Z pair is consumed
        # further down, after the Z-point measures.
        l2_d, l3_d, l2_z, l3_z = _leith_length_scales(
            grid,
            dxF=geom.dxF,
            dyF=geom.dyF,
            dxV=geom.dxV,
            dyU=geom.dyU,
            dtype=hDiv.dtype,
            device=hDiv.device,
            ny=ny,
            nx=nx,
        )

        # ── Z-point (corner) vorticity-gradient measures ──
        if seam_aware:
            # Halos from ``exchange``: the wrap face on a periodic axis of one
            # tile, the neighbour rank's face across a rank seam, zero on a wall
            # or open edge.
            vrt_ny = vrtDx.shape[-2] - 1
            vrt_nx = vrtDy.shape[-1] - 1
            vrtDx_ext = exchange(
                vrtDx[..., :, :vrt_nx],
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
            vrtDy_ext = exchange(
                vrtDy[..., :vrt_ny, :],
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
            vrt_z_max, grd_vrt_z_sq = _z_point_vorticity_gradient_measures_from_haloes(
                vrtDx_ext,
                vrtDy_ext,
                ny=vrt_ny,
                nx=vrt_nx,
                want_sq=want_div,
            )
        else:
            vrt_z_max, grd_vrt_z_sq = _z_point_vorticity_gradient_measures(
                vrtDx,
                vrtDy,
                want_sq=want_div,
            )

        # ── Z-point divergence-gradient measures (divDx j,j-1 ; divDy i,i-1) ──
        if not want_div:
            grd_div_z_max = grd_div_z_sq = None
        elif seam_aware:
            div_ny = divDy.shape[-2] - 1
            div_nx = divDx.shape[-1] - 1
            divDx_ext = exchange(
                divDx,
                grid=grid,
                location="U",
                extent=HaloExtentV1(
                    west_left_extent=0,
                    east_right_extent=0,
                    south_bottom_extent=1,
                    north_top_extent=1,
                ),
                fill=FillSpec.ZERO,
            )
            divDy_ext = exchange(
                divDy,
                grid=grid,
                location="V",
                extent=HaloExtentV1(
                    west_left_extent=1,
                    east_right_extent=1,
                    south_bottom_extent=0,
                    north_top_extent=0,
                ),
                fill=FillSpec.ZERO,
            )
            grd_div_z_max, grd_div_z_sq = _z_point_divergence_gradient_measures_from_haloes(
                divDx_ext,
                divDy_ext,
                ny=div_ny,
                nx=div_nx,
            )
        else:
            grd_div_z_max, grd_div_z_sq = _z_point_divergence_gradient_measures(
                divDx,
                divDy,
            )

        # ── Z-point Smag strain magnitude (strain primary, tension 4-avg) ──
        if seam_aware:
            tension_ny, tension_nx = tension.shape[-2:]
            tension_ext = exchange(
                tension,
                grid=grid,
                location="T",
                extent=HaloExtentV1.uniform(1),
                fill=FillSpec.ZERO,
            )
            smag_z = _z_point_smag_strain_magnitude_from_halo(
                strain, tension_ext, ny=tension_ny, nx=tension_nx,
            )
        else:
            smag_z = _z_point_smag_strain_magnitude(strain, tension)

        # (Z-point length scales l2_z/l3_z came from the Site-3 memo above.)

        return ViscosityMeasures(
            vrt_d_max=vrt_d_max, grd_vrt_d_sq=grd_vrt_d_sq,
            grd_div_d_max=grd_div_d_max, grd_div_d_sq=grd_div_d_sq,
            smag_d=smag_d, l2_d=l2_d, l3_d=l3_d,
            vrt_z_max=vrt_z_max, grd_vrt_z_sq=grd_vrt_z_sq,
            grd_div_z_max=grd_div_z_max, grd_div_z_sq=grd_div_z_sq,
            smag_z=smag_z, l2_z=l2_z, l3_z=l3_z,
            # The producer's gate decision travels WITH the measures: measures
            # built by an instance whose gate was shut cannot be fed to an
            # assembly that needs those fields (they are None).  Stamping it here
            # keeps the two sides on one predicate and makes the misuse say so.
            div_measures_present=want_div,
        )

    def _compute_leith_smag_viscosities(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        grid: "StructuredCGrid",
        terms: tuple[DivVortGeometry, torch.Tensor, DivVortInvariants] | None = None,
        measures: ViscosityMeasures | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute MITgcm-style variable harmonic viscosity ``viscAh_D`` (cell
        centres) and ``viscAh_Z`` (vorticity corners).

        Default ``use_full_leith=False`` + ``visc_c2_leith_d=0`` reproduces the
        legacy max-norm vorticity-only Leith branch byte-for-byte.  Setting
        either activates the divergence-Leith term (``leithD*grad(hDiv)``) and,
        with ``use_full_leith``, MITgcm's L2-norm sqrt branch (``mom_calc_visc.F``
        useFullLeith=.TRUE.) that combines the vorticity and divergence gradients.
        """
        if terms is None:
            side_mask_fac = self.side_drag_factor if self.no_slip_sides else 0.0
            terms = self._divvort_terms(u, v, grid, side_mask_fac=side_mask_fac)
        geom, _, inv = terms
        m = measures if measures is not None else self._leith_grad_measures(geom, inv, grid)
        return self._harmonic_from_measures(m)

    def _leith_div_active(self) -> bool:
        """True when the divergence-Leith / full-Leith assembly runs.

        SINGLE SOURCE for the predicate: read here by
        :meth:`_harmonic_from_measures` (the consumer) and, through
        :meth:`_needs_leith_sq_and_div_measures`, by
        :meth:`_leith_grad_measures` (the producer).  A second copy of this
        expression would let the producer and consumer disagree about whether the
        divergence measures are required.
        """
        return self.use_full_leith or self.visc_c2_leith_d != 0.0

    def _needs_leith_sq_and_div_measures(self) -> bool:
        """True when the divergence-gradient chain and the ``*_sq`` measures
        have a consumer.

        They are read in exactly two places: :meth:`_harmonic_from_measures`
        under :meth:`_leith_div_active`, and
        :meth:`_compute_biharmonic_viscosities`, which is reachable only under
        :meth:`_biharmonic_active`.  With both off, ``divDx``/``divDy``, both
        point-divergence measures and both ``grd_vrt_*_sq`` are computed and
        discarded ( Top-10 #1/#2/#5/#6).
        """
        return self._leith_div_active() or self._biharmonic_active()

    def _harmonic_from_measures(
        self, m: ViscosityMeasures,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        leith_div_active = self._leith_div_active()
        _require_div_measures(m, leith_div_active, "harmonic")
        if not leith_div_active:
            leith_d, leith_z = _harmonic_leith_vorticity(
                m, visc_c2_leith=self.visc_c2_leith,
            )
        elif self.use_full_leith:
            leith_d, leith_z = _harmonic_leith_full(
                m,
                visc_c2_leith=self.visc_c2_leith,
                visc_c2_leith_d=self.visc_c2_leith_d,
            )
        else:
            leith_d, leith_z = _harmonic_leith_maxnorm(
                m,
                visc_c2_leith=self.visc_c2_leith,
                visc_c2_leith_d=self.visc_c2_leith_d,
            )
        return _harmonic_viscosity_with_smag(
            leith_d, leith_z, m, visc_c2_smag=self.visc_c2_smag,
        )

    def _compute_biharmonic_viscosities(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        grid: "StructuredCGrid",
        terms: tuple[DivVortGeometry, torch.Tensor, DivVortInvariants] | None = None,
        measures: ViscosityMeasures | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MITgcm biharmonic viscosity ``viscA4_D`` / ``viscA4_Z`` (no base, no
        floor — added by :meth:`_compute_biharmonic_viscosities_with_floor`).

        Leith4 by the max-norm or the full-Leith kernel (``use_full_leith``),
        then the Smagorinsky part."""
        if terms is None:
            side_mask_fac = self.side_drag_factor if self.no_slip_sides else 0.0
            terms = self._divvort_terms(u, v, grid, side_mask_fac=side_mask_fac)
        geom, _, inv = terms
        m = measures if measures is not None else self._leith_grad_measures(geom, inv, grid)
        _require_div_measures(m, True, "biharmonic")
        l5_d, l4_d = _biharmonic_length_powers(m.l2_d)
        l5_z, l4_z = _biharmonic_length_powers(m.l2_z)
        if self.use_full_leith:
            leith4_d, leith4_z = _biharmonic_leith_full(
                m, l5_d, l5_z,
                visc_c4_leith=self.visc_c4_leith,
                visc_c4_leith_d=self.visc_c4_leith_d,
            )
        else:
            leith4_d, leith4_z = _biharmonic_leith_maxnorm(
                m, l5_d, l5_z,
                visc_c4_leith=self.visc_c4_leith,
                visc_c4_leith_d=self.visc_c4_leith_d,
            )
        return _biharmonic_viscosity_with_smag(
            leith4_d, leith4_z, m, l4_d, l4_z, visc_c4_smag=self.visc_c4_smag,
        )

    def _compute_biharmonic_viscosities_with_floor(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        grid: "StructuredCGrid",
        terms: tuple[DivVortGeometry, torch.Tensor, DivVortInvariants] | None = None,
        measures: ViscosityMeasures | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full biharmonic coefficient = base ``viscA4`` + Leith4/Smag4, with the
        ``viscA4GridMin/Max * L4rdt`` floor/cap (``mom_calc_visc.F:519-531``)."""
        viscA4_D, viscA4_Z = self._compute_biharmonic_viscosities(
            u, v, grid, terms=terms, measures=measures,
        )
        base = viscA4_D.new_tensor(self.viscosity_a4)
        viscA4_D, viscA4_Z = _add_biharmonic_base_viscosity(
            viscA4_D,
            viscA4_Z,
            base,
        )
        return self._apply_grid_visca4_floor_cap(viscA4_D, viscA4_Z, grid)

    def _biharmonic_active(self) -> bool:
        """True when any biharmonic coefficient is set (otherwise del^4 skipped,
        existing harmonic cases byte-identical)."""
        return (
            self.viscosity_a4 != 0.0
            or self.visc_c4_leith != 0.0
            or self.visc_c4_leith_d != 0.0
            or self.visc_c4_smag != 0.0
        )

    def _biharmonic_dissipation(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        grid: "StructuredCGrid",
        terms: tuple[DivVortGeometry, torch.Tensor, DivVortInvariants] | None = None,
        measures: ViscosityMeasures | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MITgcm biharmonic momentum dissipation -del^4(u,v).

        del^2(u,v) -> (dStar,zStar)=hDiv/vort3 of del^2 -> viscA4-weighted
        div-vort stencil, negated (``mom_vecinv.F`` useBiharmonicVisc chain +
        ``mom_vi_hdissip.F`` biharmonic block, ``#undef MOM_VI_ORIGINAL_VISCA4``).

        ``terms`` accepts a caller-built ``_divvort_terms(u, v, grid,
        side_mask_fac=...)`` result for the SAME ``(u, v, grid)`` and the same
        ``side_mask_fac`` expression recomputed below; ``None`` keeps the
        standalone behaviour."""
        if terms is None:
            side_mask_fac = self.side_drag_factor if self.no_slip_sides else 0.0
            terms = self._divvort_terms(u, v, grid, side_mask_fac=side_mask_fac)
        geom, hFacZ, inv = terms
        del2u, del2v = self._del2uv_fields(geom, hFacZ, inv.hDiv, inv.vort3, grid)
        # dStar = hDiv(del2u,del2v), zStar = vort3(del2u,del2v) — same routines.
        star_geom, star_hFacZ, star_inv = self._divvort_terms(del2u, del2v, grid)
        viscA4_D, viscA4_Z = self._compute_biharmonic_viscosities_with_floor(
            u, v, grid, terms=terms, measures=measures,
        )
        del4_u, del4_v = self._divvort_apply_variable(
            star_geom, star_hFacZ, star_inv.hDiv, star_inv.vort3,
            viscA4_D, viscA4_Z, grid,
        )
        if self.no_slip_sides:
            # MOM_U/V_SIDEDRAG uses Ah_Z*u - A4_Z*del2u at each
            # adjoining corner. The A4 part has the opposite drag sign.
            u_side, v_side = self._variable_side_drag(
                del2u, del2v, grid, geom, hFacZ, viscA4_Z
            )
            return -del4_u - u_side, -del4_v - v_side
        return -del4_u, -del4_v

    def _variable_side_drag(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        grid: StructuredCGrid,
        geom: DivVortGeometry,
        hFacZ: torch.Tensor,
        corner_viscosity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MOM_U/V_SIDEDRAG with the actual viscosity at each corner.

        Applied to velocity for harmonic drag and to del2 velocity for the
        biharmonic contribution, whose caller reverses the sign. Both use
        the original velocity's geometry, as in MOM_VECINV/MOM_FLUXFORM:
        the closed fraction of the resting faces and corners, the current
        ``recip_hFac`` (:func:`_side_drag_closed_fractions`).
        """
        h0FacW, h0FacS, h0FacZ = _side_drag_closed_fractions(
            grid, geom.hFacW, geom.hFacS, hFacZ,
            region_fill=FillSpec.WRAP,
        )
        dtype, device = u.dtype, u.device
        return _variable_side_drag_tendencies(
            u,
            v,
            h0FacW=h0FacW,
            h0FacS=h0FacS,
            h0FacZ=h0FacZ,
            dxV=geom.dxV,
            dyU=geom.dyU,
            recip_hFacW=geom.recip_hFacW,
            recip_hFacS=geom.recip_hFacS,
            rAw=grid.metrics.rAw_for(dtype, device),
            rAs=grid.metrics.rAs_for(dtype, device),
            corner_viscosity=corner_viscosity,
            side_drag_factor=self.side_drag_factor,
        )

    def _apply_grid_visca4_floor_cap(
        self,
        viscA4_D: torch.Tensor,
        viscA4_Z: torch.Tensor,
        grid: "StructuredCGrid",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MITgcm biharmonic grid+dt floor/cap (``mom_calc_visc.F:528-531``):

            L4rdt        = 0.03125 * L2**2 / dt   (mom_init_fixed.F:113-114, :137-138)
            viscA4_DMin  = viscA4GridMin*L4rdt   (U4scl=0, viscA4ReMax=0)
            viscA4_D     = MAX(viscA4_DMin, base+Leith4)
            viscA4_DMax  = MIN(viscA4GridMax*L4rdt, viscA4Max)
            viscA4_D     = MIN(viscA4_DMax, viscA4_D)

        with ``L2 = L2_D`` at D and ``L2_Z`` at Z points
        (``_grid_clamp_length_scales``, the same lengths the harmonic clamp
        uses).  No-op when both coefficients are 0 (existing cases unaffected)."""
        grid_min = float(self.visc_a4_grid_min)
        grid_max = float(self.visc_a4_grid_max)
        if grid_min <= 0.0 and grid_max <= 0.0:
            return viscA4_D, viscA4_Z
        if self.dt <= 0.0:
            raise ValueError(
                "horizontal_viscosity.visc_a4_grid_min/max require dt>0 "
                "(set `dt` in the horizontal_viscosity kwargs = stepper deltaT)"
            )
        l2_d, l2_z = _grid_clamp_length_scales(grid, viscA4_D.dtype, viscA4_D.device)
        l4rdt_d = _biharmonic_grid_rdt_scale(l2_d, self.dt)
        l4rdt_z = _biharmonic_grid_rdt_scale(l2_z, self.dt)
        if grid_min > 0.0:
            viscA4_D, viscA4_Z = _grid_floor(viscA4_D, viscA4_Z, l4rdt_d, l4rdt_z, grid_min)
        if grid_max > 0.0:
            viscA4_D, viscA4_Z = _grid_cap(viscA4_D, viscA4_Z, l4rdt_d, l4rdt_z, grid_max)
        return viscA4_D, viscA4_Z

    @staticmethod
    def _leith_vorticity_gradients(
        *,
        vort3: torch.Tensor,
        dxG_ext: torch.Tensor,
        dyG: torch.Tensor,
        maskS: torch.Tensor,
        maskW: torch.Tensor,
        grid: "StructuredCGrid",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return MITgcm-style Leith vorticity gradients."""
        dtype = vort3.dtype
        device = vort3.device

        maskS_eff = maskS
        maskW_eff = maskW
        # MITgcm `pkg/mom_common/mom_calc_visc.F:294-314` applies OBCS
        # interior masks to Leith vorticity gradients:
        # `vrtDx *= maskS*maskInS` and `vrtDy *= maskW*maskInW`.
        if grid.masks.maskInS is not None:
            maskInS = grid.masks.maskInS.to(dtype=dtype, device=device)
            maskS_eff = maskS_eff * maskInS.unsqueeze(0)
        if grid.masks.maskInW is not None:
            maskInW = grid.masks.maskInW.to(dtype=dtype, device=device)
            maskW_eff = maskW_eff * maskInW.unsqueeze(0)

        return _leith_vorticity_gradients_from_masks(
            vort3,
            dxG_ext=dxG_ext,
            dyG=dyG,
            maskS=maskS_eff,
            maskW=maskW_eff,
        )

    @staticmethod
    def _divvort_dissipation(
        u: torch.Tensor,
        v: torch.Tensor,
        viscosity: torch.Tensor,
        grid: "StructuredCGrid",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MITgcm div-vort horizontal viscosity from (u,v) — thin wrapper that
        builds the terms then applies the ``mom_vi_hdissip`` stencil."""
        geom, hFacZ, inv = HorizontalViscosity._divvort_terms(u, v, grid)
        return HorizontalViscosity._divvort_apply_constant(
            geom, hFacZ, inv.hDiv, inv.vort3, viscosity, grid
        )

    def _divvort_constant_biharmonic(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        viscosity: torch.Tensor,
        grid: "StructuredCGrid",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Constant div-vort viscosity with the biharmonic term (``mom_vecinv.F``).

        Harmonic part as :meth:`_divvort_dissipation`; MOM_VI_DEL2UV gives
        del2u/del2v (:400-403), their hDiv and vort3 give dStar/zStar (:404-406),
        and mom_vi_hdissip.F applies viscA4 to them and subtracts the result
        (:199-225).  The side drag acts on ``u`` with ``viscosity`` and on ``del2u``
        with ``viscosity_a4``, the latter with the opposite sign (:470-480).  The
        variable-coefficient chain of ``leith_smag`` is :meth:`_biharmonic_dissipation`.
        """
        geom, hFacZ, inv = self._divvort_terms(u, v, grid)
        u_result, v_result = self._divvort_apply_constant(
            geom, hFacZ, inv.hDiv, inv.vort3, viscosity, grid
        )
        del2u, del2v = self._del2uv_fields(geom, hFacZ, inv.hDiv, inv.vort3, grid)
        star_geom, star_hFacZ, star_inv = self._divvort_terms(del2u, del2v, grid)
        del4_u, del4_v = self._divvort_apply_constant(
            star_geom, star_hFacZ, star_inv.hDiv, star_inv.vort3,
            u.new_tensor(self.viscosity_a4), grid,
        )
        u_result = u_result - del4_u
        v_result = v_result - del4_v
        if self.no_slip_sides:
            u_result = (u_result + self._u_side_drag(u, grid, self.viscosity)
                        - self._u_side_drag(del2u, grid, self.viscosity_a4))
            v_result = (v_result + self._v_side_drag(v, grid, self.viscosity)
                        - self._v_side_drag(del2v, grid, self.viscosity_a4))
        return u_result, v_result

    @staticmethod
    def _del2uv_fields(
        geom: DivVortGeometry,
        hFacZ: torch.Tensor,
        hDiv: torch.Tensor,
        vort3: torch.Tensor,
        grid: "StructuredCGrid",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MITgcm ``MOM_VI_DEL2UV``: del^2(u,v) from (hDiv, vort3, hFacZ).

        Identical stencil to the harmonic ``mom_vi_hdissip`` application with
        unit viscosity, plus the OBCS interior mask ``maskInW``/``maskInS`` that
        ``MOM_VI_DEL2UV`` applies (``mom_vi_del2uv.F:94,119``)."""
        one = hDiv.new_tensor(1.0)
        del2u, del2v = HorizontalViscosity._divvort_apply_constant(
            geom, hFacZ, hDiv, vort3, one, grid
        )
        return _apply_del2uv_interior_masks(
            del2u,
            del2v,
            grid.masks.maskInW,
            grid.masks.maskInS,
        )

    @staticmethod
    def _divvort_apply_constant(
        geom: DivVortGeometry,
        hFacZ: torch.Tensor,
        hDiv: torch.Tensor,
        vort3: torch.Tensor,
        viscosity: torch.Tensor,
        grid: "StructuredCGrid",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Div-vort dissipation with one constant viscosity (the ``divvort``
        scheme, and the del^2 pass with unit viscosity).

        The viscosity multiplies the assembled faces: the retained faces get the
        same product as before assembly, the faces left at zero stay ``+0.0``
        for a non-negative viscosity.
        """
        hz_v3 = _divvort_corner_flux(vort3, hFacZ)
        u_dissip, v_dissip = HorizontalViscosity._divvort_stencil(geom, hDiv, hz_v3, grid)
        return viscosity * u_dissip, viscosity * v_dissip

    @staticmethod
    def _divvort_apply_variable(
        geom: DivVortGeometry,
        hFacZ: torch.Tensor,
        hDiv: torch.Tensor,
        vort3: torch.Tensor,
        visc_d: torch.Tensor,
        visc_z: torch.Tensor,
        grid: "StructuredCGrid",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Div-vort dissipation with the D/Z coefficients inside the fluxes
        (``viscAh_D/Z`` of the harmonic pass, ``viscA4_D/Z`` of del^4)."""
        hDiv_flux, hz_v3 = _divvort_variable_fluxes(hDiv, vort3, hFacZ, visc_d, visc_z)
        return HorizontalViscosity._divvort_stencil(geom, hDiv_flux, hz_v3, grid)

    @staticmethod
    def _divvort_stencil(
        geom: DivVortGeometry,
        hDiv_flux: torch.Tensor,
        hz_v3: torch.Tensor,
        grid: "StructuredCGrid",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the MITgcm div-vort dissipation stencil (``mom_vi_hdissip.F``)
        to the flux terms — used for the harmonic (hDiv,vort3,viscAh), the del^2
        (unit viscosity) and the biharmonic del^4 (dStar,zStar,viscA4) passes.
        hDivScheme=2, vort3 no-lopping.
        """
        nx = geom.nx
        ny = geom.ny
        dyG = geom.dyG
        dyC = geom.dyC
        dxC_row = geom.dxC
        dxG_ext = geom.dxG
        recip_hFacW = geom.recip_hFacW
        recip_hFacS = geom.recip_hFacS
        maskW = geom.maskW
        maskS = geom.maskS
        nb = hDiv_flux.shape[0]
        nz = hDiv_flux.shape[1]
        u = hDiv_flux.new_zeros((nb, nz, ny, nx + 1))
        v = hDiv_flux.new_zeros((nb, nz, ny + 1, nx))

        x_edge_couples = bool(grid.edge_face_couples("x"))
        y_edge_couples = bool(grid.edge_face_couples("y"))
        u_all_faces = bool(
            x_edge_couples
            or _has_distributed_neighbor(grid, "west")
            or _has_distributed_neighbor(grid, "east")
        )
        v_all_faces = bool(
            y_edge_couples
            or _has_distributed_neighbor(grid, "south")
            or _has_distributed_neighbor(grid, "north")
        )

        # ── Step 4: uDissip (mom_vi_hdissip.F:84-93) ──
        # uD2 = viscAh * (hDiv(i,j) - hDiv(i-1,j)) / dxC(i,j)
        #      - viscAh * recip_hFacW * (hFacZ(i,j+1)*vort3(i,j+1) - hFacZ(i,j)*vort3(i,j)) / dyG
        if u_all_faces:
            flux_ny, flux_nx = hDiv_flux.shape[-2:]
            hDiv_x = exchange(
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
            u_values = _u_divvort_all_from_haloed(
                hDiv_x,
                hz_v3,
                recip_hFacW,
                maskW,
                dxC_row,
                dyG,
                ny=flux_ny,
                nx=flux_nx,
            )
            u_dissip = _u_divvort_output_from_all(
                u,
                u_values,
                west=x_edge_couples or _has_distributed_neighbor(grid, "west"),
                east=x_edge_couples or _has_distributed_neighbor(grid, "east"),
            )
        else:
            u_interior = _u_divvort_interior_from_fluxes(
                hDiv_flux,
                hz_v3,
                recip_hFacW,
                maskW,
                dxC_row,
                dyG,
            )
            u_dissip = _u_divvort_output_from_interior(u, u_interior)

        # ── Step 5: vDissip (mom_vi_hdissip.F:88-93) ──
        if v_all_faces:
            flux_ny, flux_nx = hDiv_flux.shape[-2:]
            hDiv_y = exchange(
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
            v_values = _v_divvort_all_from_haloed(
                hDiv_y,
                hz_v3,
                recip_hFacS,
                maskS,
                dyC,
                dxG_ext,
                ny=flux_ny,
                nx=flux_nx,
            )
            v_dissip = _v_divvort_output_from_all(
                v,
                v_values,
                south=y_edge_couples or _has_distributed_neighbor(grid, "south"),
                north=y_edge_couples or _has_distributed_neighbor(grid, "north"),
            )
        else:
            v_interior = _v_divvort_interior_from_fluxes(
                hDiv_flux,
                hz_v3,
                recip_hFacS,
                maskS,
                dyC,
                dxG_ext,
            )
            v_dissip = _v_divvort_output_from_interior(v, v_interior)

        return u_dissip, v_dissip

    @staticmethod
    def _u_laplacian(u: torch.Tensor, grid: StructuredCGrid) -> torch.Tensor:
        """Compute ∇²u at u-face locations using MITgcm-style stencil.

        MITgcm MOM_U_DEL2U (Cartesian, free-slip):
          fZon(i,j) = hFacC(i,j) * (u(i+1,j) - u(i,j)) / dx   [at cell centers]
          fMer(i,j) = hFacZ(i,j) * (u(i,j) - u(i,j-1)) / dy   [at corners]
          del2u(i,j) = (1 / hFacW) * [(fZon(i) - fZon(i-1))/dx
                                     + (fMer(j+1) - fMer(j))/dy]

        Stages: centre flux, its x exchange, corner hFacZ (its own exchange),
        the u j-halo at coupled y edges, then the divergence.
        """
        nx = grid.nx
        ny = grid.ny
        # per-row dxC for spherical polar alignment (dy is constant).
        dxC_row_1d = grid.metrics.dxC_row_for(u.dtype, u.device)          # [ny]
        _lead = max(u.ndim - 2, 0)
        dx = dxC_row_1d.view(*([1] * _lead), ny, 1)
        dy = u.new_tensor(grid.metrics.dyG_scalar_1d)
        metrics = grid.metrics
        rAw = metrics.rAw_for(u.dtype, u.device)
        metric_aware = _strain_metric_flags(grid, dtype=u.dtype, device=u.device)[0]

        # Get hFac masks
        if grid.masks.has_3d_masks:
            hFacC = grid.masks.hFacC.to(dtype=u.dtype, device=u.device)
            hFacW = grid.masks.hFacW.to(dtype=u.dtype, device=u.device)
            if hFacC.ndim == 3:
                hFacC = hFacC.unsqueeze(0)  # [1, nz, ny, nx]
                hFacW = hFacW.unsqueeze(0)  # [1, nz, ny, nx+1]
        else:
            hFacC = torch.ones(1, 1, ny, nx, dtype=u.dtype, device=u.device)
            hFacW = torch.ones(1, 1, ny, nx + 1, dtype=u.dtype, device=u.device)

        # ── x-direction: strain flux at cell centers ──
        # fZon(i) = hFacC(i) * (u(i+1) - u(i)) / dx   for i in [0..nx-1]
        # fZon shape: [..., ny, nx]
        if metric_aware:
            dyF = metrics.dyF_for(u.dtype, u.device)
            dxF = metrics.dxF_for(u.dtype, u.device)
            fZon = _u_strain_flux_metric(u, dyF=dyF, dxF=dxF, hFacC=hFacC)
        else:
            fZon = _u_center_strain_flux(u, hFacC, dx)
        x_halo = HaloExtentV1(
            west_left_extent=1,
            east_right_extent=1,
            south_bottom_extent=0,
            north_top_extent=0,
        )
        fZon_ext = exchange(
            fZon,
            grid=grid,
            location="U",
            extent=x_halo,
            fill=FillSpec.REPLICATE,
        )

        # Divergence of fZon at u-faces [0..nx]:
        # (fZon(i) - fZon(i-1)) / dx
        if metric_aware:
            dFZon = _x_face_difference(fZon_ext)
        else:
            dFZon = _x_flux_divergence(fZon_ext, dx)  # [..., ny, nx+1]

        # ── y-direction: strain flux at corners ──
        if ny >= 2:
            # hFacZ at corners: shape [..., ny+1, nx+1]
            # MITgcm 4-neighbor formula: MIN over hFacW(j,j-1) AND hFacS(i,i-1)
            if grid.masks.has_3d_masks:
                hFacS_full = grid.masks.hFacS.to(dtype=u.dtype, device=u.device)
                if hFacS_full.ndim == 3:
                    hFacS_full = hFacS_full.unsqueeze(0)  # [1, nz, ny+1, nx]
            else:
                hFacS_full = torch.ones(1, hFacW.shape[-3], grid.ny + 1, grid.nx, dtype=u.dtype, device=u.device)
            hFacZ = _compute_hFacZ_4neighbor(hFacW, hFacS_full, grid)

            # fMer(j) = hFacZ(j) * (u(j) - u(j-1)) / dy
            # interior corners j=1..ny-1, all u-faces i; wall boundaries
            # (j=0, j=ny) keep fMer=0 for free-slip unless the edge couples.
            y_edge_couples = bool(grid.edge_face_couples("y"))
            south_couples = y_edge_couples or _has_internal_distributed_neighbor(
                grid, "south"
            )
            north_couples = y_edge_couples or _has_internal_distributed_neighbor(
                grid, "north"
            )
            u_ext = None
            if south_couples or north_couples:
                u_ext = exchange(
                    u,
                    grid=grid,
                    location="U",
                    extent=HaloExtentV1(
                        west_left_extent=0,
                        east_right_extent=0,
                        south_bottom_extent=1,
                        north_top_extent=1,
                    ),
                    fill=FillSpec.REPLICATE,
                )
            if metric_aware:
                dFMer = _u_corner_strain_divergence_metric(
                    u, u_ext,
                    dxV=metrics.dxV_for(u.dtype, u.device),
                    dyU=metrics.dyU_for(u.dtype, u.device),
                    hFacZ=hFacZ, ny=ny, nx=nx,
                    south=south_couples, north=north_couples,
                )
            else:
                if south_couples or north_couples:
                    fMer = _u_corner_strain_flux_with_edges(
                        u, u_ext, hFacZ, dy, south=south_couples, north=north_couples,
                    )
                else:
                    fMer = _u_corner_strain_flux(u, hFacZ, dy)
                # Divergence at u-faces: (fMer(j+1) - fMer(j)) / dy
                dFMer = _y_flux_divergence(fMer, dy)  # [..., ny, nx+1]
        else:
            # ny=1: no y-derivative (wall-bounded single row)
            dFMer = torch.zeros_like(dFZon)

        # ── Combine and weight by 1/hFacW ──
        x_edge_couples = bool(grid.edge_face_couples("x"))
        west_couples = x_edge_couples or _has_internal_distributed_neighbor(
            grid, "west"
        )
        east_couples = x_edge_couples or _has_internal_distributed_neighbor(
            grid, "east"
        )
        if metric_aware:
            return _u_viscous_divergence(
                dFZon, dFMer, hFacW=hFacW, rAw=rAw, nx=nx,
                west=west_couples, east=east_couples,
            )
        return _u_laplacian_row_divergence(
            dFZon, dFMer, hFacW=hFacW, nx=nx, west=west_couples, east=east_couples,
        )

    def _u_side_drag(
        self, u: torch.Tensor, grid: StructuredCGrid, viscosity: float,
    ) -> torch.Tensor:
        """No-slip side drag for u (MITgcm ``MOM_U_SIDEDRAG``), 2-D-metric form.

        Builds the per-grid corner / u-point metrics and delegates to the
        unified :func:`_side_drag_u_2d` (shared with the cubed-sphere path),
        replacing the legacy Cartesian ``dy^2`` stencil.  Returns the FULL drag
        tendency (``viscosity`` and ``sideDragFactor`` already applied); the
        biharmonic part passes ``del2u`` with ``viscosity_a4`` and its caller
        reverses the sign (``mom_u_sidedrag.F:113-140``).  Uses the
        MITgcm ``sideDragFactor>0`` branch (factor = ``side_drag_factor``, no
        extra ``×2``), so a uniform regional grid now matches MITgcm's new-form
        side drag rather than the old hard-coded ``viscFac=2`` variant.
        """
        dtype, device = u.dtype, u.device
        ny, nx = grid.ny, grid.nx
        if ny < 2:
            return torch.zeros_like(u)
        m = grid.metrics
        if grid.masks.has_3d_masks:
            hFacW = grid.masks.hFacW.to(dtype=dtype, device=device)
            hFacS = grid.masks.hFacS.to(dtype=dtype, device=device)
        else:
            nz = u.shape[-3] if u.ndim >= 3 else 1
            hFacW = torch.ones(nz, ny, nx + 1, dtype=dtype, device=device)
            hFacS = torch.ones(nz, ny + 1, nx, dtype=dtype, device=device)
        hFacZ = _compute_hFacZ_4neighbor(hFacW, hFacS, grid)        # [...,ny+1,nx+1]
        rAw = m.rAw_for(dtype, device)                               # [ny,nx+1]
        # corner metrics, full [ny+1,nx+1] (per-row, broadcast over i): the helper
        # slices dxV/dyU along BOTH axes (u uses dim -2, v uses dim -1).
        dxV = m.dxV_for(dtype, device)
        dyU = m.dyU_for(dtype, device)
        h0FacW, _, h0FacZ = _side_drag_closed_fractions(grid, hFacW, hFacS, hFacZ)
        return _side_drag_u_2d(
            u, hFacW, h0FacW, h0FacZ, rAw, dxV, dyU, viscosity, self.side_drag_factor)

    @staticmethod
    def _v_laplacian(v: torch.Tensor, grid: StructuredCGrid) -> torch.Tensor:
        """Compute ∇²v at v-face locations using MITgcm-style stencil.

        Symmetric to _u_laplacian — strain fluxes at corners (x-direction)
        and cell centers (y-direction).
        """
        nx = grid.nx
        ny = grid.ny
        # per-row dxC for spherical polar (dy constant).
        # v is on v-faces with shape (..., ny+1, nx), so dxC needs length ny+1.
        # The trailing v-face row's metric is supplied by exchange() as a
        # 1-D V-location metric row with a single north ghost cell, routing
        # the boundary fill through the Grid topology layer instead of a
        # hard ``dxC_cell[-1]`` rectangular replicate.  The north ghost entry
        # ``dxC_vface[ny]`` is sliced away by the consumer below, so the
        # ghost-fill choice does not affect the operator output.
        dxC_cell = grid.metrics.dxC_row_for(v.dtype, v.device)            # [ny]
        dxC_vface = exchange(
            dxC_cell,
            grid=grid,
            location="V",
            extent=HaloExtentV1(
                west_left_extent=0,
                east_right_extent=0,
                south_bottom_extent=0,
                north_top_extent=1,
            ),
            fill=FillSpec.REPLICATE,
        )  # [ny+1]
        _lead = max(v.ndim - 2, 0)
        dx = dxC_vface.view(*([1] * _lead), ny + 1, 1)
        dy = v.new_tensor(grid.metrics.dyG_scalar_1d)
        metrics = grid.metrics
        rAs = metrics.rAs_for(v.dtype, v.device)
        metric_aware = _strain_metric_flags(grid, dtype=v.dtype, device=v.device)[1]

        if grid.masks.has_3d_masks:
            hFacC = grid.masks.hFacC.to(dtype=v.dtype, device=v.device)
            hFacS = grid.masks.hFacS.to(dtype=v.dtype, device=v.device)
            if hFacC.ndim == 3:
                hFacC = hFacC.unsqueeze(0)  # [1, nz, ny, nx]
                hFacS = hFacS.unsqueeze(0)  # [1, nz, ny+1, nx]
        else:
            hFacC = torch.ones(1, 1, ny, nx, dtype=v.dtype, device=v.device)
            hFacS = torch.ones(1, 1, ny + 1, nx, dtype=v.dtype, device=v.device)

        # ── y-direction: strain flux at cell centers ──
        # fMer(j) = hFacC(j) * (v(j+1) - v(j)) / dy   for j in [0..ny-1]
        if metric_aware:
            dxF = metrics.dxF_for(v.dtype, v.device)
            dyF = metrics.dyF_for(v.dtype, v.device)
            fMer = _v_strain_flux_metric(v, dxF=dxF, dyF=dyF, hFacC=hFacC)
        else:
            fMer = _v_center_strain_flux(v, hFacC, dy)  # [..., ny, nx]
        y_halo = HaloExtentV1(
            west_left_extent=0,
            east_right_extent=0,
            south_bottom_extent=1,
            north_top_extent=1,
        )
        fMer_ext = exchange(
            fMer,
            grid=grid,
            location="V",
            extent=y_halo,
            fill=FillSpec.REPLICATE,
        )

        # Divergence at v-faces [0..ny]:
        if metric_aware:
            dFMer = _y_face_difference(fMer_ext)
        else:
            dFMer = _y_flux_divergence(fMer_ext, dy)  # [..., ny+1, nx]

        # ── x-direction: strain flux at corners ──
        if nx >= 2:
            # MITgcm 4-neighbor formula: MIN over hFacW(j,j-1) AND hFacS(i,i-1)
            if grid.masks.has_3d_masks:
                hFacW_full = grid.masks.hFacW.to(dtype=v.dtype, device=v.device)
                if hFacW_full.ndim == 3:
                    hFacW_full = hFacW_full.unsqueeze(0)  # [1, nz, ny, nx+1]
            else:
                hFacW_full = torch.ones(1, hFacS.shape[-3], ny, nx + 1, dtype=v.dtype, device=v.device)
            hFacZ = _compute_hFacZ_4neighbor(hFacW_full, hFacS, grid)

            x_edge_couples = bool(grid.edge_face_couples("x"))
            west_couples = x_edge_couples or _has_internal_distributed_neighbor(
                grid, "west"
            )
            east_couples = x_edge_couples or _has_internal_distributed_neighbor(
                grid, "east"
            )
            v_ext = None
            if west_couples or east_couples:
                v_ext = exchange(
                    v,
                    grid=grid,
                    location="V",
                    extent=HaloExtentV1(
                        west_left_extent=1,
                        east_right_extent=1,
                        south_bottom_extent=0,
                        north_top_extent=0,
                    ),
                    fill=FillSpec.REPLICATE,
                )
            if metric_aware:
                dFZon = _v_corner_strain_divergence_metric(
                    v, v_ext,
                    dxV=metrics.dxV_for(v.dtype, v.device),
                    dyU=metrics.dyU_for(v.dtype, v.device),
                    hFacZ=hFacZ, ny=ny, nx=nx,
                    west=west_couples, east=east_couples,
                )
            else:
                if west_couples or east_couples:
                    fZon = _v_corner_strain_flux_with_edges(
                        v, v_ext, hFacZ, dx, west=west_couples, east=east_couples,
                    )
                else:
                    fZon = _v_corner_strain_flux(v, hFacZ, dx)
                # Divergence: (fZon(i+1) - fZon(i)) / dx at v-faces
                dFZon = _x_flux_divergence(fZon, dx)  # [..., ny+1, nx]
        else:
            dFZon = torch.zeros_like(dFMer)

        # ── Combine and weight by 1/hFacS ──
        y_edge_couples = bool(grid.edge_face_couples("y"))
        south_couples = y_edge_couples or _has_internal_distributed_neighbor(
            grid, "south"
        )
        north_couples = y_edge_couples or _has_internal_distributed_neighbor(
            grid, "north"
        )
        if metric_aware:
            return _v_viscous_divergence(
                dFZon, dFMer, hFacS=hFacS, rAs=rAs, ny=ny,
                south=south_couples, north=north_couples,
            )
        return _v_laplacian_row_divergence(
            dFMer, dFZon, hFacS=hFacS, ny=ny, south=south_couples, north=north_couples,
        )

    def _v_side_drag(
        self, v: torch.Tensor, grid: StructuredCGrid, viscosity: float,
    ) -> torch.Tensor:
        """No-slip side drag for v (MITgcm ``MOM_V_SIDEDRAG``), 2-D-metric form.

        Symmetric to :meth:`_u_side_drag`; delegates to the unified
        :func:`_side_drag_v_2d`.  Returns the FULL drag (viscosity + factor
        applied).
        """
        dtype, device = v.dtype, v.device
        ny, nx = grid.ny, grid.nx
        if nx < 2:
            return torch.zeros_like(v)
        m = grid.metrics
        if grid.masks.has_3d_masks:
            hFacW = grid.masks.hFacW.to(dtype=dtype, device=device)
            hFacS = grid.masks.hFacS.to(dtype=dtype, device=device)
        else:
            nz = v.shape[-3] if v.ndim >= 3 else 1
            hFacW = torch.ones(nz, ny, nx + 1, dtype=dtype, device=device)
            hFacS = torch.ones(nz, ny + 1, nx, dtype=dtype, device=device)
        hFacZ = _compute_hFacZ_4neighbor(hFacW, hFacS, grid)        # [...,ny+1,nx+1]
        rAs = m.rAs_for(dtype, device)                               # [ny+1,nx]
        dxV = m.dxV_for(dtype, device)
        dyU = m.dyU_for(dtype, device)
        _, h0FacS, h0FacZ = _side_drag_closed_fractions(grid, hFacW, hFacS, hFacZ)
        return _side_drag_v_2d(
            v, hFacS, h0FacS, h0FacZ, rAs, dyU, dxV, viscosity, self.side_drag_factor)

    def _strain_tension_variable_dissipation(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        grid: StructuredCGrid,
        diagnostics: DiagnosticState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MITgcm's strain-tension harmonic viscosity with the variable coefficients.

        ``mom_vecinv.F`` with useStrainTensionVisc computes the coefficients as for
        the div-vort form (MOM_CALC_VISC on hDiv, vort3BC, tension, strainBC,
        :362-381), zeroes strain at dry corners (:338-341), applies MOM_HDISSIP with
        ``viscAh_Z`` on strain and ``viscAh_D`` on tension (:419-427), then adds the
        side drag (:470-480).  Stages: invariants and coefficients, the tension
        fluxes' x and y halos, the stencil, the side drag.  An edge face is kept
        only where the edge couples, as in :meth:`_divvort_stencil`.
        """
        side_mask_fac = self.side_drag_factor if self.no_slip_sides else 0.0
        terms = self._divvort_terms(u, v, grid, side_mask_fac=side_mask_fac)
        geom, hFacZ, inv = terms
        measures = self._leith_grad_measures(geom, inv, grid)
        viscAh_D, viscAh_Z = self._compute_leith_smag_viscosities_with_floor(
            u, v, grid, terms=terms, measures=measures,
        )
        diagnostics.set_tensor("viscAh_W", _w_viscosity_from_center_viscosity(viscAh_D))
        diagnostics.set_tensor("viscAh3d_00", diagnostics.tensors["viscAh_W"])

        # strainBC equals strain on wet corners, so selecting zero on the dry ones
        # gives MITgcm's masked strain.
        strain = torch.where(
            hFacZ.unsqueeze(0) == 0, torch.zeros_like(inv.strain_bc), inv.strain_bc,
        )
        u_center = _strain_tension_center_flux(inv.tension, viscAh_D, length=geom.dyF)
        v_center = _strain_tension_center_flux(inv.tension, viscAh_D, length=geom.dxF)
        u_center_x = exchange(
            u_center,
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
        v_center_y = exchange(
            v_center,
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
        u_values = _u_strain_tension_dissipation(
            u_center_x,
            _strain_tension_corner_flux(strain, viscAh_Z, length=geom.dxV),
            recip_dyG=_recip(geom.dyG),
            recip_dxC=_recip(geom.dxC),
        )
        v_values = _v_strain_tension_dissipation(
            _strain_tension_corner_flux(strain, viscAh_Z, length=geom.dyU),
            v_center_y,
            recip_dyC=_recip(geom.dyC),
            recip_dxG=_recip(geom.dxG),
        )
        x_edge_couples = bool(grid.edge_face_couples("x"))
        y_edge_couples = bool(grid.edge_face_couples("y"))
        u_result = _u_divvort_output_from_all(
            u,
            u_values,
            west=x_edge_couples or _has_distributed_neighbor(grid, "west"),
            east=x_edge_couples or _has_distributed_neighbor(grid, "east"),
        )
        v_result = _v_divvort_output_from_all(
            v,
            v_values,
            south=y_edge_couples or _has_distributed_neighbor(grid, "south"),
            north=y_edge_couples or _has_distributed_neighbor(grid, "north"),
        )
        if self.no_slip_sides:
            u_side, v_side = self._variable_side_drag(u, v, grid, geom, hFacZ, viscAh_Z)
            u_result = u_result + u_side
            v_result = v_result + v_side
        return u_result, v_result

    def _flux_form_variable_dissipation(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        grid: StructuredCGrid,
        diagnostics: DiagnosticState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MITgcm's flux-form harmonic viscosity with the variable coefficients.

        ``mom_fluxform.F`` computes hDiv, vort3, tension and strain and scales
        vort3 and strain by sideMaskFac at dry corners (:330-352; the
        vector-invariant vort3BC / strainBC of ``_divvort_terms``), calls
        MOM_CALC_VISC for viscAh_D / viscAh_Z (:453-458), hands them to
        MOM_U/V_XVISCFLUX and YVISCFLUX (:612-618, :907-913) and the flux
        divergence (:628-644, :923-939), and to the side drag (:656-669,
        :951-964).  The coefficients are those of the ``leith_smag`` scheme; only
        the stencil they enter differs.  A biharmonic coefficient adds viscA4_D /
        viscA4_Z from the same MOM_CALC_VISC (:meth:`_flux_form_biharmonic`).
        """
        side_mask_fac = self.side_drag_factor if self.no_slip_sides else 0.0
        terms = self._divvort_terms(u, v, grid, side_mask_fac=side_mask_fac)
        geom, hFacZ, inv = terms
        measures = self._leith_grad_measures(geom, inv, grid)
        viscAh_D, viscAh_Z = self._compute_leith_smag_viscosities_with_floor(
            u, v, grid, terms=terms, measures=measures,
        )
        diagnostics.set_tensor("viscAh_W", _w_viscosity_from_center_viscosity(viscAh_D))
        diagnostics.set_tensor("viscAh3d_00", diagnostics.tensors["viscAh_W"])
        if self._biharmonic_active():
            viscA4_D, viscA4_Z = self._compute_biharmonic_viscosities_with_floor(
                u, v, grid, terms=terms, measures=measures,
            )
            return self._flux_form_biharmonic(
                u, v, grid, geom, hFacZ, viscAh_D, viscAh_Z, viscA4_D, viscA4_Z,
            )
        u_result = self._u_flux_viscous(u, grid, geom, hFacZ, viscAh_D, viscAh_Z)
        v_result = self._v_flux_viscous(v, grid, geom, hFacZ, viscAh_D, viscAh_Z)
        if self.no_slip_sides:
            u_side, v_side = self._variable_side_drag(u, v, grid, geom, hFacZ, viscAh_Z)
            u_result = u_result + u_side
            v_result = v_result + v_side
        return u_result, v_result

    @staticmethod
    def _u_flux_viscous(
        u: torch.Tensor,
        grid: StructuredCGrid,
        geom: DivVortGeometry,
        hFacZ: torch.Tensor,
        visc_d: torch.Tensor,
        visc_z: torch.Tensor,
    ) -> torch.Tensor:
        """Harmonic flux-form dissipation of u with viscAh_D / viscAh_Z.

        ``mom_u_xviscflux.F:54-71`` / ``mom_u_yviscflux.F:55-76`` (harmonic part;
        cosFacU = 1 for cosPower = 0, ini_spherical_polar_grid.F:256-268) and
        ``mom_fluxform.F:628-644``, with drF cancelled and MITgcm's two minus signs
        folded::

            fZon(i,j) = dyF hFacC viscAh_D (u(i+1,j) - u(i,j)) / dxF     C points
            fMer(i,j) = dxV hFacZ viscAh_Z (u(i,j) - u(i,j-1)) / dyU     Z points
            guDiss    = recip_hFacW recip_rAw ((fZon(i)-fZon(i-1)) + (fMer(j+1)-fMer(j)))

        The edge faces and corners are those of the metric branch of
        :meth:`_u_laplacian`: a coupled seam (periodic or rank) takes halo values,
        an uncoupled edge face the replicated zonal difference (zero).
        """
        nx, ny = grid.nx, grid.ny
        dtype, device = u.dtype, u.device
        rAw = grid.metrics.rAw_for(dtype, device)
        fZon = _u_viscous_zonal_flux(
            u, dyF=geom.dyF, dxF=geom.dxF, hFacC=geom.hFacC, visc_d=visc_d,
        )
        fZon_ext = exchange(
            fZon,
            grid=grid,
            location="U",
            extent=HaloExtentV1(west_left_extent=1, east_right_extent=1,
                                south_bottom_extent=0, north_top_extent=0),
            fill=FillSpec.REPLICATE,
        )
        dFZon = _x_face_difference(fZon_ext)                         # [..., ny, nx+1]
        if ny >= 2:
            y_edge_couples = bool(grid.edge_face_couples("y"))
            south = y_edge_couples or _has_internal_distributed_neighbor(grid, "south")
            north = y_edge_couples or _has_internal_distributed_neighbor(grid, "north")
            u_ext = None
            if south or north:
                u_ext = exchange(
                    u,
                    grid=grid,
                    location="U",
                    extent=HaloExtentV1(west_left_extent=0, east_right_extent=0,
                                        south_bottom_extent=1, north_top_extent=1),
                    fill=FillSpec.REPLICATE,
                )
            dFMer = _u_viscous_meridional_divergence(
                u, u_ext, dxV=geom.dxV, dyU=geom.dyU, hFacZ=hFacZ, visc_z=visc_z,
                ny=ny, nx=nx, south=south, north=north,
            )                                                       # [..., ny, nx+1]
        else:
            dFMer = torch.zeros_like(dFZon)
        x_edge_couples = bool(grid.edge_face_couples("x"))
        west = x_edge_couples or _has_internal_distributed_neighbor(grid, "west")
        east = x_edge_couples or _has_internal_distributed_neighbor(grid, "east")
        return _u_viscous_divergence(
            dFZon, dFMer, hFacW=geom.hFacW, rAw=rAw, nx=nx, west=west, east=east,
        )

    @staticmethod
    def _v_flux_viscous(
        v: torch.Tensor,
        grid: StructuredCGrid,
        geom: DivVortGeometry,
        hFacZ: torch.Tensor,
        visc_d: torch.Tensor,
        visc_z: torch.Tensor,
    ) -> torch.Tensor:
        """Harmonic flux-form dissipation of v, symmetric to :meth:`_u_flux_viscous`.

        ``mom_v_xviscflux.F:55-72`` / ``mom_v_yviscflux.F:54-76`` (cosFacV = 1) and
        ``mom_fluxform.F:923-939``::

            fZon(i,j) = dyU hFacZ viscAh_Z (v(i,j) - v(i-1,j)) / dxV     Z points
            fMer(i,j) = dxF hFacC viscAh_D (v(i,j+1) - v(i,j)) / dyF     C points
            gvDiss    = recip_hFacS recip_rAs ((fZon(i+1)-fZon(i)) + (fMer(j)-fMer(j-1)))
        """
        nx, ny = grid.nx, grid.ny
        dtype, device = v.dtype, v.device
        rAs = grid.metrics.rAs_for(dtype, device)
        fMer = _v_viscous_meridional_flux(
            v, dxF=geom.dxF, dyF=geom.dyF, hFacC=geom.hFacC, visc_d=visc_d,
        )
        fMer_ext = exchange(
            fMer,
            grid=grid,
            location="V",
            extent=HaloExtentV1(west_left_extent=0, east_right_extent=0,
                                south_bottom_extent=1, north_top_extent=1),
            fill=FillSpec.REPLICATE,
        )
        dFMer = _y_face_difference(fMer_ext)                         # [..., ny+1, nx]
        if nx >= 2:
            x_edge_couples = bool(grid.edge_face_couples("x"))
            west = x_edge_couples or _has_internal_distributed_neighbor(grid, "west")
            east = x_edge_couples or _has_internal_distributed_neighbor(grid, "east")
            v_ext = None
            if west or east:
                v_ext = exchange(
                    v,
                    grid=grid,
                    location="V",
                    extent=HaloExtentV1(west_left_extent=1, east_right_extent=1,
                                        south_bottom_extent=0, north_top_extent=0),
                    fill=FillSpec.REPLICATE,
                )
            dFZon = _v_viscous_zonal_divergence(
                v, v_ext, dxV=geom.dxV, dyU=geom.dyU, hFacZ=hFacZ, visc_z=visc_z,
                ny=ny, nx=nx, west=west, east=east,
            )                                                       # [..., ny+1, nx]
        else:
            dFZon = torch.zeros_like(dFMer)
        y_edge_couples = bool(grid.edge_face_couples("y"))
        south = y_edge_couples or _has_internal_distributed_neighbor(grid, "south")
        north = y_edge_couples or _has_internal_distributed_neighbor(grid, "north")
        return _v_viscous_divergence(
            dFZon, dFMer, hFacS=geom.hFacS, rAs=rAs, ny=ny, south=south, north=north,
        )

    def _flux_form_constant_biharmonic(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        grid: StructuredCGrid,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Constant flux-form viscosity with the biharmonic term.

        MITgcm fills viscAh_D/Z and viscA4_D/Z with the constants
        (``mom_fluxform.F:444-452``) and runs the fluxes of the variable case, so
        the constants go to :meth:`_flux_form_biharmonic` as tensors.  No grid floor
        or cap: MITgcm applies them in MOM_CALC_VISC only (:453-458).  The corner
        ``hFacZ`` is MOM_CALC_HFACZ's four-neighbour MIN, as in the variable case.
        """
        dtype, device = u.dtype, u.device
        geom = divvort_geometry(grid, nz=u.shape[-3], dtype=dtype, device=device)
        hFacZ = _divvort_corner_hfacz(
            grid, geom.hFacW, geom.hFacS,
            HaloExtentV1(west_left_extent=0, east_right_extent=0,
                         south_bottom_extent=1, north_top_extent=1),
            HaloExtentV1(west_left_extent=1, east_right_extent=1,
                         south_bottom_extent=0, north_top_extent=0),
            dtype, device,
        )
        center = (*u.shape[:-2], grid.ny, grid.nx)
        corner = (*u.shape[:-2], grid.ny + 1, grid.nx + 1)
        return self._flux_form_biharmonic(
            u, v, grid, geom, hFacZ,
            u.new_full(center, self.viscosity), u.new_full(corner, self.viscosity),
            u.new_full(center, self.viscosity_a4), u.new_full(corner, self.viscosity_a4),
        )

    def _flux_form_biharmonic(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        grid: StructuredCGrid,
        geom: DivVortGeometry,
        hFacZ: torch.Tensor,
        viscAh_D: torch.Tensor,
        viscAh_Z: torch.Tensor,
        viscA4_D: torch.Tensor,
        viscA4_Z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MITgcm flux-form horizontal viscosity with the biharmonic term.

        ``mom_fluxform.F:601-669`` for u, :897-964 for v: MOM_U/V_DEL2U give
        del2u/del2v (main term times ``maskW*maskInW``, then the no-slip term times
        ``maskW``, ``mom_u_del2u.F:88-121``); the viscous fluxes carry
        ``viscAh*d(u) - viscA4*d(del2u)``; the side drag acts on ``u`` with viscAh_Z
        and on ``del2u`` with viscA4_Z, the latter with the opposite sign
        (``mom_u_sidedrag.F``).  The no-slip term and the side drag read the resting
        closed fractions of :func:`_side_drag_closed_fractions`.
        """
        dtype, device = u.dtype, u.device
        del2u, u_ext = self._u_del2_flux_form(u, grid, geom, hFacZ)
        del2v, v_ext = self._v_del2_flux_form(v, grid, geom, hFacZ)
        del2u, del2v = _apply_del2uv_interior_masks(
            del2u, del2v, grid.masks.maskInW, grid.masks.maskInS,
        )
        if self.no_slip_sides:
            h0FacW, h0FacS, h0FacZ = _side_drag_closed_fractions(
                grid, geom.hFacW, geom.hFacS, hFacZ, region_fill=FillSpec.WRAP,
            )
            del2u = del2u + _u_del2_side_term(
                u, hFacW=geom.hFacW, h0FacW=h0FacW, h0FacZ=h0FacZ,
                rAw=grid.metrics.rAw_for(dtype, device), dxV=geom.dxV, dyU=geom.dyU,
                maskW=geom.maskW, side_drag_factor=self.side_drag_factor,
            )
            del2v = del2v + _v_del2_side_term(
                v, hFacS=geom.hFacS, h0FacS=h0FacS, h0FacZ=h0FacZ,
                rAs=grid.metrics.rAs_for(dtype, device), dyU=geom.dyU, dxV=geom.dxV,
                maskS=geom.maskS, side_drag_factor=self.side_drag_factor,
            )
        u_result = self._u_flux_viscous_biharmonic(
            u, u_ext, del2u, grid, geom, hFacZ, viscAh_D, viscAh_Z, viscA4_D, viscA4_Z,
        )
        v_result = self._v_flux_viscous_biharmonic(
            v, v_ext, del2v, grid, geom, hFacZ, viscAh_D, viscAh_Z, viscA4_D, viscA4_Z,
        )
        if self.no_slip_sides:
            u_side, v_side = self._variable_side_drag(u, v, grid, geom, hFacZ, viscAh_Z)
            u_side4, v_side4 = self._variable_side_drag(
                del2u, del2v, grid, geom, hFacZ, viscA4_Z,
            )
            u_result = u_result + u_side - u_side4
            v_result = v_result + v_side - v_side4
        return u_result, v_result

    @staticmethod
    def _u_del2_flux_form(
        u: torch.Tensor,
        grid: StructuredCGrid,
        geom: DivVortGeometry,
        hFacZ: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """MOM_U_DEL2U main term times ``maskW`` (``mom_u_del2u.F:53-96``).

        Exchanges, in order: the zonal flux (U, one i-halo column each side,
        replicated), then ``u`` (U, one j-halo row each side) when a y edge couples.
        Returns del2u and that ``u`` halo (``None`` when no y edge couples), which
        the viscous fluxes reuse.
        """
        nx, ny = grid.nx, grid.ny
        rAw = grid.metrics.rAw_for(u.dtype, u.device)
        fZon = _u_del2_zonal_flux(
            u, dyF=geom.dyF, dxF=geom.dxF, hFacC=geom.hFacC, maskInC=geom.maskInC,
        )
        fZon_ext = exchange(
            fZon,
            grid=grid,
            location="U",
            extent=HaloExtentV1(west_left_extent=1, east_right_extent=1,
                                south_bottom_extent=0, north_top_extent=0),
            fill=FillSpec.REPLICATE,
        )
        dFZon = _x_face_difference(fZon_ext)                         # [..., ny, nx+1]
        u_ext = None
        if ny >= 2:
            y_edge_couples = bool(grid.edge_face_couples("y"))
            south = y_edge_couples or _has_internal_distributed_neighbor(grid, "south")
            north = y_edge_couples or _has_internal_distributed_neighbor(grid, "north")
            if south or north:
                u_ext = exchange(
                    u,
                    grid=grid,
                    location="U",
                    extent=HaloExtentV1(west_left_extent=0, east_right_extent=0,
                                        south_bottom_extent=1, north_top_extent=1),
                    fill=FillSpec.REPLICATE,
                )
            dFMer = _u_corner_strain_divergence_metric(
                u, u_ext, dxV=geom.dxV, dyU=geom.dyU, hFacZ=hFacZ,
                ny=ny, nx=nx, south=south, north=north,
            )                                                       # [..., ny, nx+1]
        else:
            dFMer = torch.zeros_like(dFZon)
        x_edge_couples = bool(grid.edge_face_couples("x"))
        west = x_edge_couples or _has_internal_distributed_neighbor(grid, "west")
        east = x_edge_couples or _has_internal_distributed_neighbor(grid, "east")
        del2u = _u_del2(
            dFZon, dFMer, hFacW=geom.hFacW, rAw=rAw, maskW=geom.maskW,
            nx=nx, west=west, east=east,
        )
        return del2u, u_ext

    @staticmethod
    def _v_del2_flux_form(
        v: torch.Tensor,
        grid: StructuredCGrid,
        geom: DivVortGeometry,
        hFacZ: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """MOM_V_DEL2V main term times ``maskS`` (``mom_v_del2v.F:52-95``).

        Exchanges, in order: the meridional flux (V, one j-halo row each side,
        replicated), then ``v`` (V, one i-halo column each side) when an x edge
        couples.  Returns del2v and that ``v`` halo.
        """
        nx, ny = grid.nx, grid.ny
        rAs = grid.metrics.rAs_for(v.dtype, v.device)
        fMer = _v_del2_meridional_flux(
            v, dxF=geom.dxF, dyF=geom.dyF, hFacC=geom.hFacC, maskInC=geom.maskInC,
        )
        fMer_ext = exchange(
            fMer,
            grid=grid,
            location="V",
            extent=HaloExtentV1(west_left_extent=0, east_right_extent=0,
                                south_bottom_extent=1, north_top_extent=1),
            fill=FillSpec.REPLICATE,
        )
        dFMer = _y_face_difference(fMer_ext)                         # [..., ny+1, nx]
        v_ext = None
        if nx >= 2:
            x_edge_couples = bool(grid.edge_face_couples("x"))
            west = x_edge_couples or _has_internal_distributed_neighbor(grid, "west")
            east = x_edge_couples or _has_internal_distributed_neighbor(grid, "east")
            if west or east:
                v_ext = exchange(
                    v,
                    grid=grid,
                    location="V",
                    extent=HaloExtentV1(west_left_extent=1, east_right_extent=1,
                                        south_bottom_extent=0, north_top_extent=0),
                    fill=FillSpec.REPLICATE,
                )
            dFZon = _v_corner_strain_divergence_metric(
                v, v_ext, dxV=geom.dxV, dyU=geom.dyU, hFacZ=hFacZ,
                ny=ny, nx=nx, west=west, east=east,
            )                                                       # [..., ny+1, nx]
        else:
            dFZon = torch.zeros_like(dFMer)
        y_edge_couples = bool(grid.edge_face_couples("y"))
        south = y_edge_couples or _has_internal_distributed_neighbor(grid, "south")
        north = y_edge_couples or _has_internal_distributed_neighbor(grid, "north")
        del2v = _v_del2(
            dFZon, dFMer, hFacS=geom.hFacS, rAs=rAs, maskS=geom.maskS,
            ny=ny, south=south, north=north,
        )
        return del2v, v_ext

    @staticmethod
    def _u_flux_viscous_biharmonic(
        u: torch.Tensor,
        u_ext: torch.Tensor | None,
        del2u: torch.Tensor,
        grid: StructuredCGrid,
        geom: DivVortGeometry,
        hFacZ: torch.Tensor,
        visc_d: torch.Tensor,
        visc_z: torch.Tensor,
        visc4_d: torch.Tensor,
        visc4_z: torch.Tensor,
    ) -> torch.Tensor:
        """Flux-form dissipation of u with the harmonic and biharmonic terms
        (``mom_u_xviscflux.F:56-67``, ``mom_u_yviscflux.F:57-72``,
        ``mom_fluxform.F:628-644``), edges as in :meth:`_u_flux_viscous`.

        Exchanges, in order: the zonal flux (U, x), then ``del2u`` (U, one j-halo
        row each side) when a y edge couples; the ``u`` halo is the one
        :meth:`_u_del2_flux_form` exchanged.
        """
        nx, ny = grid.nx, grid.ny
        rAw = grid.metrics.rAw_for(u.dtype, u.device)
        fZon = _u_viscous_zonal_flux_biharmonic(
            u, del2u, dyF=geom.dyF, dxF=geom.dxF, hFacC=geom.hFacC,
            visc_d=visc_d, visc4_d=visc4_d,
        )
        fZon_ext = exchange(
            fZon,
            grid=grid,
            location="U",
            extent=HaloExtentV1(west_left_extent=1, east_right_extent=1,
                                south_bottom_extent=0, north_top_extent=0),
            fill=FillSpec.REPLICATE,
        )
        dFZon = _x_face_difference(fZon_ext)                         # [..., ny, nx+1]
        if ny >= 2:
            y_edge_couples = bool(grid.edge_face_couples("y"))
            south = y_edge_couples or _has_internal_distributed_neighbor(grid, "south")
            north = y_edge_couples or _has_internal_distributed_neighbor(grid, "north")
            del2u_ext = None
            if south or north:
                del2u_ext = exchange(
                    del2u,
                    grid=grid,
                    location="U",
                    extent=HaloExtentV1(west_left_extent=0, east_right_extent=0,
                                        south_bottom_extent=1, north_top_extent=1),
                    fill=FillSpec.REPLICATE,
                )
            dFMer = _u_viscous_meridional_divergence_biharmonic(
                u, u_ext, del2u, del2u_ext, dxV=geom.dxV, dyU=geom.dyU, hFacZ=hFacZ,
                visc_z=visc_z, visc4_z=visc4_z, ny=ny, nx=nx, south=south, north=north,
            )                                                       # [..., ny, nx+1]
        else:
            dFMer = torch.zeros_like(dFZon)
        x_edge_couples = bool(grid.edge_face_couples("x"))
        west = x_edge_couples or _has_internal_distributed_neighbor(grid, "west")
        east = x_edge_couples or _has_internal_distributed_neighbor(grid, "east")
        return _u_viscous_divergence(
            dFZon, dFMer, hFacW=geom.hFacW, rAw=rAw, nx=nx, west=west, east=east,
        )

    @staticmethod
    def _v_flux_viscous_biharmonic(
        v: torch.Tensor,
        v_ext: torch.Tensor | None,
        del2v: torch.Tensor,
        grid: StructuredCGrid,
        geom: DivVortGeometry,
        hFacZ: torch.Tensor,
        visc_d: torch.Tensor,
        visc_z: torch.Tensor,
        visc4_d: torch.Tensor,
        visc4_z: torch.Tensor,
    ) -> torch.Tensor:
        """Flux-form dissipation of v with the harmonic and biharmonic terms
        (``mom_v_xviscflux.F:57-68``, ``mom_v_yviscflux.F:56-71``,
        ``mom_fluxform.F:923-939``), symmetric to :meth:`_u_flux_viscous_biharmonic`.

        Exchanges, in order: the meridional flux (V, y), then ``del2v`` (V, one
        i-halo column each side) when an x edge couples.
        """
        nx, ny = grid.nx, grid.ny
        rAs = grid.metrics.rAs_for(v.dtype, v.device)
        fMer = _v_viscous_meridional_flux_biharmonic(
            v, del2v, dxF=geom.dxF, dyF=geom.dyF, hFacC=geom.hFacC,
            visc_d=visc_d, visc4_d=visc4_d,
        )
        fMer_ext = exchange(
            fMer,
            grid=grid,
            location="V",
            extent=HaloExtentV1(west_left_extent=0, east_right_extent=0,
                                south_bottom_extent=1, north_top_extent=1),
            fill=FillSpec.REPLICATE,
        )
        dFMer = _y_face_difference(fMer_ext)                         # [..., ny+1, nx]
        if nx >= 2:
            x_edge_couples = bool(grid.edge_face_couples("x"))
            west = x_edge_couples or _has_internal_distributed_neighbor(grid, "west")
            east = x_edge_couples or _has_internal_distributed_neighbor(grid, "east")
            del2v_ext = None
            if west or east:
                del2v_ext = exchange(
                    del2v,
                    grid=grid,
                    location="V",
                    extent=HaloExtentV1(west_left_extent=1, east_right_extent=1,
                                        south_bottom_extent=0, north_top_extent=0),
                    fill=FillSpec.REPLICATE,
                )
            dFZon = _v_viscous_zonal_divergence_biharmonic(
                v, v_ext, del2v, del2v_ext, dxV=geom.dxV, dyU=geom.dyU, hFacZ=hFacZ,
                visc_z=visc_z, visc4_z=visc4_z, ny=ny, nx=nx, west=west, east=east,
            )                                                       # [..., ny+1, nx]
        else:
            dFZon = torch.zeros_like(dFMer)
        y_edge_couples = bool(grid.edge_face_couples("y"))
        south = y_edge_couples or _has_internal_distributed_neighbor(grid, "south")
        north = y_edge_couples or _has_internal_distributed_neighbor(grid, "north")
        return _v_viscous_divergence(
            dFZon, dFMer, hFacS=geom.hFacS, rAs=rAs, ny=ny, south=south, north=north,
        )
