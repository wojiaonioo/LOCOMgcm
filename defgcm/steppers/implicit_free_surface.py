"""Implicit free surface stepper with preconditioned conjugate gradient solver.

Solves the 2D Helmholtz equation for η^(n+1) each time step, eliminating
barotropic CFL constraints. Following MITgcm's semi-implicit approach:

1. Evaluate slow tendencies and apply Adams-Bashforth extrapolation
2. Predict u* = u^n + dt * (AB2_tendency - (1-α)·g·∇η^n)
2b. Implicit vertical viscosity: (I - dt·K·∂²/∂z²) u* = u*_predict
3. Assemble RHS: etaH - β·dt·div(transport*)
4. Solve: [-α·β·g·dt_mom·dt_freesurf·∇·(H·∇) + I] η^{n+1} = RHS
5. Correct: u^{n+1} = u* - α·g·dt·∇η^{n+1}, masked by maskInW
6. Recompute η from corrected velocity divergence (exactConserv)

Parameters α (implicSurfPress) and β (implicDiv2DFlow) control the
implicit/explicit splitting. Default α=β=0.5 matches MITgcm's internal_wave
test case. Adams-Bashforth (ab_eps=0.1) damps 2Δt oscillations for
semi-implicit schemes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import dataclasses
import logging
import os
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Mapping, Optional

import torch

from defgcm.dycore_parameters import NonhydrostaticParams
from defgcm.dynamics import AssemblyPolicy
from defgcm.dynamics.assembly_policy import AssemblyBucket
from defgcm.dynamics.base import MomentumTermClass
from defgcm.dynamics.cd_scheme import admit_cd_scheme, cd_scheme_of, decode_cd_state, encode_cd_state
from defgcm.seaice.step_state import decode_seaice_state, encode_seaice_state
from defgcm.dynamics.convective_adjustment import convective_adjustment_of
from defgcm.dynamics.equation_of_state import zero_negative_salinity
from defgcm.dynamics.vertical_momentum import compute_gw
from defgcm.dynamics.vertical_mixing_schemes.coefficients import bottom_interface_viscosity
from defgcm.dynamics.vertical_mixing_schemes.density_gradient import vertical_density_gradient
from defgcm.dynamics.implicit_vertical_advection import tracer_hfac_factors
from defgcm.solvers.implicit_vertical import TracerWNpHalfMIT
from defgcm.operators.fluxes.tracer_flux import TracerFluxScheme
from defgcm.constants import ROTATION_PERIOD_EARTH, omega_from_rotation_period
from defgcm.dynamics.kpp_mixing import KppStatic, subcycle_kpp_tracers
from defgcm.accel.graph_exit import graph_exit
from defgcm.geometry import StructuredCGrid
from defgcm.geometry.grid import _compute_interior_mask, ensure_grid_device_resident
from defgcm.diagnostics import ckpt_dump
from defgcm.solvers import (
    timestep_wvel,
)
from defgcm.state import DiagnosticState, EvaluationInputs, PrognosticState, StateTendency
from defgcm.step.contract import EMPTY_MAPPING, StepDiagnostics, StepOutcome

from .base import TimeStepBundle, TimeStepper, coerce_time_step_bundle, tracer_boundary_phases
from .ifs import admission, prepare
from .split_explicit import SplitStepperModelContext
from .ifs import coefficients, columns, pressure
from .ifs.carry import _carry, cast_tensor
from .ifs.trace import StreamTrace, _dist_trace, record_checkpoint, track_surface_solution
from .ifs import coriolis, surface as ifs_surface
from .ifs.cube_inputs import CubeSurfaceForcing, EMPTY_CUBE_SURFACE_FORCING
from .ifs.kpp_inputs import (
    KppEvaluation,
    KppRunValues,
    KppStaticInputs,
    KppStepForcing,
    kpp_eos_module,
    kpp_eos_name,
    kpp_eos_type,
    prepare_kpp_face_stress,
    prepare_kpp_heat_salt_forcing,
    prepare_kpp_static,
    resolve_kpp_coriolis,
    prepare_kpp_surface_forcing,
    prepare_kpp_wind_forcing,
    prepare_regional_kpp_inputs,
    prepare_regional_kpp_surface_inputs,
)
from .ifs import tendency as ifs_tendency
from .ifs import tracer_stage as ifs_tracer
from .ifs.freshwater import ContinuityHistory
from .ifs import freshwater as ifs_freshwater
from .ifs import publication as ifs_publication
from .ifs import nlfs_geometry as ifs_geometry
from .ifs import nh as ifs_nh
from .ifs import r_star
from .ifs import state as ifs_state
from .step_segments import (
    make_momentum_predictor_masks, surface_pressure_gradients, apply_explicit_surface_pressure,
    mask_velocity_pair, reciprocal_velocity_hfacs, mask_surface_pressure_gradients,
    surface_pressure_face_masks,
    compute_free_surface_rhs, prepare_free_surface_solve,
    subtract_real_freshwater_rhs, _solve_cg2d_and_stage,
    compute_velocity_correction,
    compute_velocity_predictor,
    compute_unmasked_velocity_predictor, apply_unmasked_explicit_surface_pressure,
)
from defgcm.steppers._registry import register_stepper

if TYPE_CHECKING:
    from defgcm.dynamics.surface_flux import SurfaceForcing
    from defgcm.step.contract import BoundaryPayload, StepPlan

_LOG = logging.getLogger(__name__)

#: Free-surface tracking printf (diagnostic, default OFF); evaluate once at import to avoid graph breaks from environment reads in the compiled step body.
_R4_TRACK_ENABLED = os.environ.get("R4_TRACK") == "1"




from .implicit_free_surface_cg2d import (
    _assemble_helmholtz_coefficients,
    cg2d_w_unit_tolerance,
)
from .implicit_free_surface_cube import (
    CubeStaticState,
    INITIAL_CUBE_STATIC_STATE,
    step_cube as _step_cube_impl,
)
from .implicit_free_surface_nonlin import (
    apply_surface_eta_boundary, _rescale_tendency_nonlin_freesurf,
)




# ---------------------------------------------------------------------------
# Implicit Free Surface Stepper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
@register_stepper("implicit_free_surface")
class ImplicitFreeSurfaceStepper(TimeStepper):
    supports_fullgraph_model_step: ClassVar[bool] = True
    # Only the slow (AB2) tendency is evaluated here; a FAST-domain module is
    # refused at assembly rather than dropped.  PREDICTOR modules (the CD
    # scheme) are called in _prepare_momentum_prediction, TRACER_CORRECTION
    # modules (convective adjustment) in _finalize_regional_state.
    evaluated_domains: frozenset[str] = frozenset({"slow", "predictor", "tracer_correction"})
    # The default module list carries the explicit barotropic continuity and
    # pressure-gradient terms for the explicit steppers; this stepper solves
    # the free surface implicitly instead, so their absence from the evaluated
    # domains is the design, not a dropped term.
    superseded_module_names: frozenset[str] = frozenset({"continuity", "pressure_gradient"})
    """Semi-implicit free surface time stepper.

    Eliminates barotropic CFL constraint by solving a 2D Helmholtz equation
    for the free surface height at each time step. Allows dt >> sqrt(gH)/dx.

    Parameters
    ----------
    implicit_surf_press : float
        Fraction of surface pressure gradient treated implicitly (α).
        The explicit fraction (1-α) is added to the prediction step.
        Default 0.5 matches MITgcm's internal_wave test case.
    implicit_div_2d_flow : float
        Fraction of 2D flow divergence treated implicitly (β).
        Default 0.5 matches MITgcm's internal_wave test case.
    mom_stepping : bool
        MITgcm ``momStepping`` (default true, set_defaults.F:192).  ``False``
        runs the tracers in a given flow: FORWARD_STEP then skips DYNAMICS,
        SOLVE_FOR_PRESSURE and MOMENTUM_CORRECTION_STEP (forward_step.F:786-795,
        :897-903, :911-917), so u, v and eta keep their values, no u* is
        predicted, no CG2D solve runs and the total hydrostatic potential
        history keeps its initial value.
    temp_stepping, salt_stepping : bool
        MITgcm ``tempStepping``, ``saltStepping`` (default true,
        set_defaults.F:194, :198).  ``False`` skips TEMP_INTEGRATE or
        SALT_INTEGRATE (thermodynamics.F:317-337): the tracer keeps its value
        through the explicit update, Adams-Bashforth and every column solve;
        the boundary application that follows still acts on it, as
        OBCS_APPLY_TS does on both tracers (thermodynamics.F:356-360).
    momentum_pressure_forcing : bool
        MITgcm ``momPressureForcing`` (``pfFacMom``, set_parms.F:223-227).
        ``False`` keeps the free-surface solve but applies neither the
        explicit nor the implicit surface pressure gradient to u/v
        (timestep.F:80, correction_step.F:152). Retain the hydrostatic
        pressure module: it still produces phiHyd for initialization and
        pressure history, while its momentum gradient is disabled.
    ab_eps : float
        Adams-Bashforth damping parameter (MITgcm abEps, whose default is 0.01,
        set_defaults.F:317).  The tendency used is G + (0.5 + ab_eps)(G - G_prev),
        i.e. weights (1.5 + ab_eps) and -(0.5 + ab_eps) (adams_bashforth2.F:64).
        Set to 0 to disable AB2 (forward Euler for the tendencies, gW included);
        MITgcm runs that scheme with abEps = -0.5 (abFac = 0), not with
        abEps = ab_eps, so the MIT exporter rejects ab_eps <= 0.
        Extrapolated tendency = (1.5 + ab_eps)*G^n - (0.5 + ab_eps)*G^{n-1}
    exact_conserv : bool
        If True, recompute η from corrected velocity divergence after
        the correction step, matching MITgcm's exactConserv=.TRUE.
        This ensures η is exactly consistent with the volume transport.
    use_real_freshwater_flux : bool
        If True, include MITgcm's real-freshwater volume flux in the integrated
        CG2D RHS and exactConserv etaH evolution.  Default False matches
        ``set_defaults.F:263`` and preserves manifests that omit the gate.
    nonlin_free_surf : int
        Nonlinear free surface level (MITgcm: nonlinFreeSurf).
    vector_invariant_momentum : bool or None
        MITgcm ``vectorInvariantMomentum``, read only by the ``nonlin_free_surf
        >= 2`` momentum tendency rescale, which MITgcm takes outside the
        vector-invariant form only (``timestep.F:275-276``).  A vector-invariant
        or flux-form core in the composition names the form itself (the
        declaration must then agree); a composition without such a core (a bare
        ``FCoriolis``, MITgcm ``momAdvection=.FALSE.``) needs the declaration
        when ``nonlin_free_surf >= 2`` (``resolve_vector_invariant_momentum``).
    hfac_inf : float
        MITgcm ``hFacInf``: lower bound on the surface-cell thickness fraction
        under a nonlinear free surface (inert at ``nonlin_free_surf = 0``).
        0 = linear free surface (default).
        >=2 = rescale momentum tendency at surface cells by
        hFacW_old / hFac_surfW to account for thickness changes.
    select_r_star : int
        MITgcm ``select_rStar``: 0 moves only the surface level
        (CALC/UPDATE_SURF_DR), 2 the r* coordinate, every level by the column
        factor (CALC/UPDATE_R_STAR), with ``nonlin_free_surf = 4``.
    """
    # Topology capability (defgcm/grid_support.py).
    # The ONLY stepper with the face_count==6 cube dispatch + seam-aware
    # exchange; the model's cube-stepper guard reads this declaration.
    # Class attribute (no annotation) so the frozen dataclass is unaffected
    # and IFS subclasses inherit the claim, preserving the isinstance
    # semantics of the original guard.
    grid_support = ("latlon", "cube")

    OWNED_STEP_FIELDS: ClassVar[dict] = {
        # The whole-model nonlinear_geometry codec saves this owner.
        "_surface_geometry": (None, None),
        # The same codec owns the r* factors.
        "_r_star": (None, None),
        "_previous_eta": (
            ifs_state.encode_previous_eta, ifs_state.decode_previous_eta,
        ),
        "_kpp_static_inputs": (None, None),
        "_nh_history": (
            ifs_nh.encode_nh_history, ifs_nh.decode_nh_history,
        ),
        "_continuity": (
            ifs_freshwater.encode_continuity_history, ifs_freshwater.decode_continuity_history,
        ),
        "_pmep_r": (
            ifs_freshwater.encode_freshwater_history, ifs_freshwater.decode_freshwater_history,
        ),
        "_tracer_ab": (
            ifs_tracer.encode_tracer_ab_history, ifs_tracer.decode_tracer_ab_history,
        ),
        "_momentum_ab": (
            ifs_tendency.encode_momentum_ab_history, ifs_tendency.decode_momentum_ab_history,
        ),
        "_step_index": (
            ifs_state.encode_step_index, ifs_state.decode_step_index,
        ),
        "_obc_eta": (
            ifs_state.encode_boundary_eta, ifs_state.decode_boundary_eta,
        ),
        "_cd_state": (encode_cd_state, decode_cd_state),
        "_ggl90_tke": (columns.encode_ggl90_tke, columns.decode_ggl90_tke),
        "_seaice": (encode_seaice_state, decode_seaice_state),
    }

    gravity: float = 9.81
    implicit_surf_press: float = 0.5
    implicit_div_2d_flow: float = 0.5
    momentum_pressure_forcing: bool = True
    mom_stepping: bool = True
    temp_stepping: bool = True
    salt_stepping: bool = True
    # Explicit CALC_GW gates. U/V module composition is declared separately;
    # absence of a momentum module does not define these W-equation inputs.
    nh_mom_advection: bool = True
    nh_mom_viscosity: bool = True
    ab_eps: float = 0.1
    exact_conserv: bool = True
    use_real_freshwater_flux: bool = False
    implicit_viscosity: float = 0.0
    implicit_vertical_viscosity: bool = True
    implicit_tracer_diffusion: bool = False
    temperature_implicit_vertical_advection: bool = False
    salinity_implicit_vertical_advection: bool = False
    momentum_implicit_vertical_advection: bool = False
    upwind_shear: bool = False
    tracer_vertical_diffusivity_kr: float = 0.0
    # MITgcm ivdc_kappa: vertical diffusivity added on statically unstable
    # interfaces of tracers without KPP (calc_3d_diffusivity.F:107-116).
    ivdc_kappa: float = 0.0
    # MITgcm usePP81 / useMY82 / useGGL90: one of columns.VERTICAL_MIXING_SCHEMES.  The
    # parameters are the scheme's namelist values by record field name; the
    # construction replaces the mapping with the scheme's record (KL10: the
    # KLviscMax scalar).
    vertical_mixing_scheme: str = "none"
    vertical_mixing_parameters: Any = None
    # cube horizontal tracer diffusion scalar coefficient (m2/s): feeds the cube
    # inline GAD horizontal diffusion; operator is an is_cube leaf.  Default
    # 0.0 == cube's old fallback (no horizontal diffusion).  Single source of truth
    # is physics.diffusivity, injected into stepper.kwargs by the cube runtime.
    tracer_diffusivity_kh: float = 0.0
    nonlin_free_surf: int = 0
    # MITgcm hFacInf (set_defaults.F:258).  Lower bound on the surface cell
    # thickness under a nonlinear free surface; inert when nonlin_free_surf = 0.
    hfac_inf: float = 0.2
    select_r_star: int = 0
    # MITgcm vectorInvariantMomentum; None: named by the composition's momentum
    # core (resolve_vector_invariant_momentum, at model assembly).
    vector_invariant_momentum: bool | None = None
    # MITgcm gad_init_fixed.F:143-163 derives these independently from each
    # tracer's stepping flag and horizontal advection scheme.  ``None`` is not
    # a physical default: callers must provide both fields or the explicit
    # legacy spelling below.
    ab2_on_temperature_tendency: bool | None = None
    ab2_on_salinity_tendency: bool | None = None
    # Backward-compatible explicit spelling.  It has no default physics:
    # omission without the two fields above raises, and an explicit pair takes
    # precedence when audit-oriented manifests carry both representations.
    ab2_on_tracer_tendency: bool | None = None
    stagger_tracer_stepping: bool = False
    # MITgcm momDissip_In_AB — when False (e.g. exp4 verification),
    # viscous dissipation (HorizontalViscosity + VerticalMixing momentum
    # part) is excluded from the AB2 extrapolation history (guNm1 in
    # MITgcm) and added AFTER the extrapolation each step, mirroring
    # timestep.F:128-136 vs 219-226. Default True matches MITgcm global
    # default (set_defaults.F:315) and the internal-wave case.
    mom_dissip_in_ab: bool = True
    # MITgcm momForcingOutAB — when 1, momentum forcing is excluded from
    # guNm1/gvNm1 and added AFTER AB2; when 0 it enters the AB2 history
    # (timestep.F:160-185).  This has no behavioural default: each run must
    # declare the regime selected by its MITgcm reference configuration.
    mom_forcing_out_ab: bool | None = None
    # Explicit policy input preserving the eight existing cases that place
    # only bottom drag after AB2 while other dissipation remains in AB2.
    bottom_drag_in_ab: bool = True
    # Optional cube-specific AssemblyPolicy inputs.  None selects the explicit
    # cube-neutral defaults in AssemblyPolicy.cube_neutral; a partial mapping
    # overrides only named policy inputs, and an AssemblyPolicy replaces all.
    cube_assembly_policy: AssemblyPolicy | Mapping[str, bool] | None = None
    # MITgcm implicitIntGravWave — together with staggerTimeStep this gates
    # the HPG AB2 placement.  HPG enters AB2 only when both flags are False
    # (synchronous + non-IIGW branch, timestep.F:116-126).  Otherwise HPG is
    # added AFTER AB2 (staggered/IIGW branch, timestep.F:320-325).  The internal-wave
    # and exp4 verification cases both have implicitIntGravWave=F, so the default here
    # selects the HPG-in-AB2 path when stagger_tracer_stepping=False.
    implicit_int_grav_wave: bool = False
    # MITgcm CHECK_SALINITY_FOR_NEGATIVE_VALUES (CPP_OPTIONS.h, undefined by
    # default): FIND_RHO_2D resets negative prognostic salt to zero in place
    # (find_rho.F:85-89).  The salt-changing calls are the step's rhoInSitu
    # (do_oceanic_phys.F:758-765) and the convective adjustment's densities
    # (convective_adjustment.F:123-136); admission refuses the compositions
    # whose other calls or readers of the unreset salt are not ported.
    check_salinity_for_negative_values: bool = False
    cg_max_iters: int = 1000
    cg_tolerance: float = 1e-13
    # MITgcm's cg2dTargetResidual is a dimensionless test on ||r|| / max|rhs|
    # (cg2d.F).  The historical LOCOMgcm default compared ||r|| in absolute
    # units against the same 1e-13, which fp64 cannot reach at production scale:
    # every solve then ran until the pAp breakdown guard fired (55-58 iterations,
    # true residual ~6e-6) while MITgcm stopped at 36.
    # The default is aligned with MITgcm so the same cg_tolerance value means
    # the same test in both models.  Standalone configs
    # that need the old criterion set cg_tolerance_mode="absolute" explicitly.
    # "mit_w_unit" is MITgcm's cg2dTargetResWunit > 0 test with cg_tolerance in
    # m/s (implicit_free_surface_cg2d.CG2D_STOP_CRITERIA, cg2d_w_unit_tolerance).
    cg_tolerance_mode: str = "mit_normalized"
    # The CG3D mode is separate because only the regional NH branch consumes it.
    # Default absolute preserves the pre-operator-resolved solver path.
    cg3d_tolerance_mode: str = "absolute"
    # MITgcm cg2dUseMinResSol: return the CG2D iterate of lowest residual when
    # the solve ends unconverged (cg2d.F:358-364).  MITgcm's default depends on
    # the grid and topography files (ini_parms.F:1585-1588); a translated
    # configuration writes the resolved value.
    cg_use_min_res_sol: bool = False
    # An external non-hydrostatic phi_nh provider adds the correction
    # `-dt*grad(phi_nh)` to U and V
    # at the same stage MITgcm applies cg3d correction (correction_step.F:
    # 153-165), after the 2D barotropic eta correction and before
    # enforce_boundaries / integr_continuity. Provider signature:
    # f(time_seconds: float) -> Optional[Tensor [nz, ny, nx]]. It is called
    # with the end-of-step time = time_seconds + dt. Returning None or leaving
    # provider unset disables this path. W is diagnostic from continuity, so it
    # is not corrected here; corrected U/V feed the existing continuity path.
    external_pnh_provider: Optional[Callable[[float], Optional[torch.Tensor]]] = None
    nonhydrostatic: NonhydrostaticParams = NonhydrostaticParams()
    # KPP vertical mixing (Large et al. 1994).  When True the step diagnoses
    # KPP diffusivities from the current state each step and applies them as the
    # implicit vertical viscosity / tracer diffusion + ghat nonlocal transport
    # (defgcm.dynamics.kpp_mixing).  Cube path also honours the cache flag
    # ``_cube_kpp_enabled`` (harness-set).  Default False -> baseline unchanged.
    kpp_enabled: bool = False
    # Effective physical mode for raw forcing adapters; a SurfaceForcing bundle
    # carries its producer's mode for both the tracer and KPP consumers.
    penetrating_sw: bool = False
    # Regional staggered T/S substeps refresh KPP from evolving tracers while
    # retaining the outer momentum/forcing clock. One preserves the reference.
    kpp_tracer_substeps: int = 1
    # Run-scoped data.kpp / KPP_OPTIONS overrides. None preserves the stock Ricr,
    # grid-derived minKPPhbl, and grid/cache-derived Coriolis paths byte-for-byte;
    # the booleans below match the package CPP defaults.
    kpp_ricr: float | None = None
    kpp_min_hbl: float | None = None
    kpp_coriolis: float | None = None
    # Optional override of the cell-centre Coriolis profile (ny local rows) for
    # KPP's Ekman depth, e.g. the "$coriolis_profile_c" sentinel or a yaml list,
    # both sliced to the rank in runtime_kwargs.  MITgcm hands KPPMIX fCori at the
    # cell centre (kpp_calc.F:555, ini_cori.F:84-95).  None: when the momentum
    # profile is the spherical map of a spherical-polar grid, prepare_kpp_static
    # takes 2*Omega*sin of the grid's centre latitudes (MetricSet.yC_row), else
    # the corner average of the momentum profile, which is the centre value for
    # a profile linear in y (f-plane, beta plane).
    kpp_coriolis_profile_c: tuple[float, ...] | None = None
    # MITgcm rotationPeriod [s] (set_defaults.F:115).  KPP's spherical f, and the
    # check that the momentum's corner profile is the spherical map, use its
    # omega (ini_parms.F:487-490).  The canonical assembly sets it from
    # physics.rotation_period; admission refuses a momentum module or
    # NonhydrostaticParams with another period.
    rotation_period: float = ROTATION_PERIOD_EARTH
    kpp_estimate_uref: bool = False
    kpp_smooth_shsq: bool = True
    kpp_smooth_dbloc: bool = True
    # Fortran SIGN(a, -0.0) of the reference MITgcm build, which decides whether
    # zero surface forcing is stable in bldepth: "signbit" (gfortran default,
    # ifx/ifort -assume minus0) or "positive" (ifx/ifort default, gfortran
    # -fno-sign-zero); see defgcm.dynamics.kpp.KPP_SIGN_ZERO_MODES and
    # docs/NUMERICS.md.
    kpp_sign_zero: str = "signbit"
    # Surface-relative buoyancy behind KPP's Ritop: "statekpp" (STATEKPP's
    # dbsfc, MITgcm) or "divided_difference" (the JMD95 density difference at
    # the level pressure, one rounding; kpp.kpp_surface_buoyancy_divided_difference).
    kpp_surface_buoyancy: str = "statekpp"
    # Arithmetic of STATEKPP's adjacent-layer density difference behind dbloc
    # (defgcm.dynamics.kpp.KPP_DBLOC_FORMS): "statekpp" subtracts two densities as
    # MITgcm STATEKPP does; "divided_difference" (JMD95 only) is the polynomial
    # divided difference, which keeps near-neutral derivatives
    # accurate.  The KPP CUDA STATEKPP and fused kernels serve "divided_difference"
    # only and decline the MITgcm form; kpp_convective_implicit requires it.
    kpp_dbloc_form: str = "statekpp"
    # MITgcm KPPuseDoubleDiff (data.kpp, default .FALSE.): salt-fingering and
    # diffusive-convection diffusivities join KPP's background (kpp.kpp_double_diffusion).
    kpp_use_double_diff: bool = False
    # MITgcm KPP_ghatUseTotalDiffus (data.kpp, default .FALSE.): with GM/Redi on,
    # KPP's nonlocal flux multiplies KPPdiffKz + Kwz (_kpp_nonlocal_diffusivities).
    kpp_ghat_use_total_diffus: bool = False
    # None: KPP coefficients lagged at the step start (MITgcm). A mapping
    # selects convective_be_column_v1, the convective coefficients solved
    # implicitly with the mixed T/S (defgcm/dynamics/kpp_convective_implicit.py):
    # {root_policy: newton | newton_rank_one_brent_v1 | newton_factor_box_v1
    #  | newton_factor_homotopy_v1,
    #  root_tolerance, polish_steps,
    #  residual_scale: {temp: [nz], salt: [nz]}, execution: eager | compiled}.
    # Resolved to a ConvectiveImplicitConfig in __post_init__.
    kpp_convective_implicit: Any = None
    # Cube tracer DST3-FL (MITgcm scheme 33, horizontal + vertical R-pass).  Read
    # via a dual-read at the cube tracer call site (cache knob
    # ``_cube_tracer_dst3fl`` / this field / env ``DEFGCM_CUBE_TRACER_DST3FL``).
    # Default True: the cube runs DST3-FL like the region + MITgcm production
    # (tempAdvScheme=33 + tempVertAdvScheme=33) instead of 1st-order upwind.
    # Justified by: op-level FP-floor vs MIT (horizontal 3.81e-12 + vertical
    # fVerT 1.14e-12), multi-card 2-rank == 1-rank bitwise, and a cs768
    # high-res forward bounded/non-growing
    # through ~1000 steps.  Set False (or env=0) for the upwind byte-id baseline.
    cube_tracer_dst3fl: bool = True
    # Cube hydrostatic-pressure-gradient gate (a regular stepper field; the yaml
    # value lives in model.stepper.kwargs).  Default True.
    # The HPG kernel itself is a standalone cube function.
    cube_enable_hpg: bool = True
    # Cube JMD95P (dynamic_pressure) gate (a regular stepper field).  True is refused by
    # validate_dynamic_pressure_composition (equation_of_state.py) — JMD95P needs the lagged
    # totPhiHyd feedback + Picard seed, neither wired for the cube.  The
    # regional JMD95P face stays model.dynamics jmd95_equation_of_state
    # kwargs.dynamic_pressure; the two are not merged.
    cube_eos_dynamic_pressure: bool = False
    # Freezing-point clamp (MITgcm allowFreezing / FREEZE_SURFACE semantics).
    # When True, the surface temperature entering the bounded state is clamped
    # to >= freezing_point (theta = max(theta, Tfreezing)); the finalized
    # endpoint remains unconstrained until the next step.  Default False ->
    # baseline / MIT-alignment runs are bitwise unchanged.  freezing_point
    # default -1.9 degC matches MITgcm's source-coded Tfreezing.
    allow_freezing: bool = False
    freezing_point: float = -1.9
    name: str = "implicit_free_surface"

    _uses_pressure_history: bool = field(default=True, init=False, repr=False)
    stream_trace: StreamTrace = field(
        default_factory=StreamTrace, init=False, repr=False, compare=False,
    )
    _resolved_assembly_policy: AssemblyPolicy | None = field(
        default=None, init=False, repr=False, compare=False,
    )
    # vectorInvariantMomentum of the assembled composition
    # (resolve_vector_invariant_momentum); None until the model assembles it.
    _resolved_vector_invariant_momentum: bool | None = field(
        default=None, init=False, repr=False, compare=False,
    )
    # The composition's package modules (admission.resolve_package_modules); None
    # until the model assembles it.
    _package_modules: admission.PackageModules | None = field(
        default=None, init=False, repr=False, compare=False,
    )
    if TYPE_CHECKING:
        _cube_surface_forcing: CubeSurfaceForcing
        cube_static_state: CubeStaticState
        cube_w_interface_readout: torch.Tensor | None
    # ``_evaluated_w_interface`` and ``_evaluated_total_phi_hyd`` (the last
    # evaluated step's lagged w and total hydrostatic potential, set by
    # ``_evaluation_inputs`` and read only by ``evaluated_inputs``) are not
    # fields: an attribute that exists when the step is traced is guarded on
    # its value, and its first write (None to tensor) recompiled the step.
    # They are absent until the first step and again after each segment.

    def __post_init__(self) -> None:
        # Runtime histories and readouts are not frozen method-definition
        # fields. Only consumers of an owner inspect its runtime values.
        object.__setattr__(self, "_cube_surface_forcing", EMPTY_CUBE_SURFACE_FORCING)
        object.__setattr__(self, "cube_static_state", INITIAL_CUBE_STATIC_STATE)
        object.__setattr__(self, "cube_w_interface_readout", None)
        object.__setattr__(self, "_kpp_static_inputs", KppStaticInputs(
            static=None, rho_const=None,
        ))
        object.__setattr__(self, "_kpp_forcing_inputs", KppStepForcing(
            surf_forc_t=None, surf_forc_s=None, surf_forc_u=None, surf_forc_v=None,
            surf_qsw=None, penetrating_sw=self.penetrating_sw,
        ))
        object.__setattr__(self, "_tracer_ab", None)
        object.__setattr__(self, "_momentum_ab", None)
        object.__setattr__(self, "_continuity", ContinuityHistory())
        object.__setattr__(self, "_surface_geometry", ifs_geometry.SurfaceGeometry())
        object.__setattr__(self, "_r_star", ifs_geometry.RStarGeometry())
        object.__setattr__(self, "_nh_history", ifs_nh.NHHistory())
        object.__setattr__(self, "_previous_eta", ifs_state.PreviousEtaHistory(None, False))
        object.__setattr__(self, "_pmep_r", None)
        object.__setattr__(self, "_step_index", None)
        object.__setattr__(self, "_obc_eta", None)
        object.__setattr__(self, "_cd_state", None)
        # The sea-ice component's carry after the last step; None is a fresh
        # run, which starts from the component's initial carry.
        object.__setattr__(self, "_seaice", None)
        # GGL90TKE after the last step; None is a fresh run, seeded by
        # prepare_step_host (ggl90_init_varia.F:42-56).
        object.__setattr__(self, "_ggl90_tke", None)
        # A runtime history, not a frozen method-definition field. Only the
        # dynamic-pressure consumer should guard this value; static-pressure
        # runs still publish it for diagnostics and checkpoint compatibility.
        object.__setattr__(self, "_cached_total_phi_hyd", None)
        # MIT SET_PARMS resolves this dependency before timestep placement.
        if not self.momentum_pressure_forcing:
            object.__setattr__(self, "implicit_int_grav_wave", False)
        convective = admission.validate_implicit_free_surface_configuration(
            kpp_tracer_substeps=self.kpp_tracer_substeps,
            kpp_enabled=self.kpp_enabled,
            stagger_tracer_stepping=self.stagger_tracer_stepping,
            select_r_star=self.select_r_star,
            nh_mom_advection=self.nh_mom_advection,
            nh_mom_viscosity=self.nh_mom_viscosity,
            temperature_implicit_vertical_advection=self.temperature_implicit_vertical_advection,
            salinity_implicit_vertical_advection=self.salinity_implicit_vertical_advection,
            momentum_implicit_vertical_advection=self.momentum_implicit_vertical_advection,
            implicit_vertical_viscosity=self.implicit_vertical_viscosity,
            use_real_freshwater_flux=self.use_real_freshwater_flux,
            upwind_shear=self.upwind_shear,
            kpp_estimate_uref=self.kpp_estimate_uref,
            kpp_smooth_shsq=self.kpp_smooth_shsq,
            kpp_smooth_dbloc=self.kpp_smooth_dbloc,
            kpp_sign_zero=self.kpp_sign_zero,
            kpp_surface_buoyancy=self.kpp_surface_buoyancy,
            kpp_dbloc_form=self.kpp_dbloc_form,
            kpp_convective_implicit=self.kpp_convective_implicit,
        )
        object.__setattr__(self, "kpp_convective_implicit", convective)
        admission.validate_implicit_free_surface_assembly(
            convective=convective,
            kpp_enabled=self.kpp_enabled,
            stagger_tracer_stepping=self.stagger_tracer_stepping,
            kpp_tracer_substeps=self.kpp_tracer_substeps,
            temperature_implicit_vertical_advection=self.temperature_implicit_vertical_advection,
            salinity_implicit_vertical_advection=self.salinity_implicit_vertical_advection,
            kpp_dbloc_form=self.kpp_dbloc_form,
            nonlin_free_surf=self.nonlin_free_surf,
            kpp_coriolis_profile_c=self.kpp_coriolis_profile_c,
            kpp_sign_zero=self.kpp_sign_zero,
            cube_assembly_policy=self.cube_assembly_policy,
            mom_forcing_out_ab=self.mom_forcing_out_ab,
        )
        admission.validate_r_star_configuration(
            select_r_star=self.select_r_star,
            nonlin_free_surf=self.nonlin_free_surf,
            mom_stepping=self.mom_stepping,
            use_real_freshwater_flux=self.use_real_freshwater_flux,
            kpp_enabled=self.kpp_enabled,
            temperature_implicit_vertical_advection=self.temperature_implicit_vertical_advection,
            salinity_implicit_vertical_advection=self.salinity_implicit_vertical_advection,
            momentum_implicit_vertical_advection=self.momentum_implicit_vertical_advection,
            implicit_div_2d_flow=self.implicit_div_2d_flow,
            nonhydrostatic=getattr(self.nonhydrostatic, "nonhydrostatic", False),
        )
        columns.validate_vertical_mixing(
            ivdc_kappa=self.ivdc_kappa, implicit_tracer_diffusion=self.implicit_tracer_diffusion,
            scheme=self.vertical_mixing_scheme, kpp_enabled=self.kpp_enabled,
            implicit_vertical_viscosity=self.implicit_vertical_viscosity,
            nonhydrostatic=getattr(self.nonhydrostatic, "nonhydrostatic", False),
            momentum_implicit_vertical_advection=self.momentum_implicit_vertical_advection,
        )
        object.__setattr__(self, "vertical_mixing_parameters", columns.vertical_mixing_parameters(
            self.vertical_mixing_scheme, self.vertical_mixing_parameters))
        temperature_ab, salinity_ab = admission.resolve_tracer_ab_flags(
            ab2_on_tracer_tendency=self.ab2_on_tracer_tendency,
            ab2_on_temperature_tendency=self.ab2_on_temperature_tendency,
            ab2_on_salinity_tendency=self.ab2_on_salinity_tendency,
        )
        object.__setattr__(self, "ab2_on_temperature_tendency", temperature_ab)
        object.__setattr__(self, "ab2_on_salinity_tendency", salinity_ab)
        admission.validate_solver_tolerance_modes(
            cg_tolerance_mode=self.cg_tolerance_mode,
            cg3d_tolerance_mode=self.cg3d_tolerance_mode,
        )
        admission._check_nonhydrostatic_implicit_factors(
            nonhydrostatic=getattr(self.nonhydrostatic, "nonhydrostatic", False),
            implicit_surf_press=self.implicit_surf_press,
            implicit_div_2d_flow=self.implicit_div_2d_flow,
        )
        admission._refuse_unimplemented_pairs(
            kpp_enabled=self.kpp_enabled,
            implicit_vertical_viscosity=self.implicit_vertical_viscosity,
            implicit_div_2d_flow=self.implicit_div_2d_flow,
            exact_conserv=self.exact_conserv,
            nonlin_free_surf=self.nonlin_free_surf,
            nonhydrostatic=getattr(self.nonhydrostatic, "nonhydrostatic", False),
        )
        object.__setattr__(self, "_kpp_run_values", KppRunValues(
            gravity=self.gravity,
            implicit_viscosity=self.implicit_viscosity, min_hbl=self.kpp_min_hbl,
            estimate_uref=self.kpp_estimate_uref,
            smooth_shsq=self.kpp_smooth_shsq, smooth_dbloc=self.kpp_smooth_dbloc,
            sign_zero=self.kpp_sign_zero, surface_buoyancy=self.kpp_surface_buoyancy,
            dbloc_form=self.kpp_dbloc_form, ricr=self.kpp_ricr,
            use_double_diff=self.kpp_use_double_diff,
        ))

    def _implicit_vertical_advection_flags(self) -> admission.ImplicitVerticalAdvectionFlags:
        return admission.ImplicitVerticalAdvectionFlags(
            temperature=self.temperature_implicit_vertical_advection,
            salinity=self.salinity_implicit_vertical_advection,
            momentum=self.momentum_implicit_vertical_advection,
        )

    def _implicit_vertical_advection_composition(
        self, model,
    ) -> admission.ImplicitVerticalAdvectionComposition:
        grid = getattr(model, "grid", None)
        if grid is None or not hasattr(grid, "face_count"):
            grid_kind = None
        else:
            grid_kind = "cube" if grid.face_count == 6 else "latlon"
        nonhydrostatic_params = getattr(self, "nonhydrostatic", None)
        nonhydrostatic = getattr(nonhydrostatic_params, "nonhydrostatic", None)
        return admission.ImplicitVerticalAdvectionComposition(
            flags=self._implicit_vertical_advection_flags(),
            stepper_name=self.name,
            stagger_tracer_stepping=self.stagger_tracer_stepping,
            nonlin_free_surf=self.nonlin_free_surf,
            select_r_star=self.select_r_star,
            nonhydrostatic=nonhydrostatic,
            grid_kind=grid_kind,
            dynamics_modules=getattr(model, "dynamics_modules", None),
        )

    def _implicit_vertical_advection_execution(
        self, model,
    ) -> admission.ImplicitVerticalAdvectionExecution:
        populate_module_diagnostics = getattr(model, "_populate_module_diagnostics", None)
        accumulate_tendency_diagnostics = getattr(model, "accumulate_tendency_diagnostics", None)
        boundary = getattr(model, "boundary_condition", None)
        enforce_tendency = getattr(boundary, "enforce_tendency", None)
        missing_required_slow_modules = []
        if self.momentum_implicit_vertical_advection and not hasattr(
            model, "slow_momentum_modules"
        ):
            missing_required_slow_modules.append("slow_momentum_modules")
        if (
            self.temperature_implicit_vertical_advection
            or self.salinity_implicit_vertical_advection
        ) and not hasattr(model, "slow_tracer_modules"):
            missing_required_slow_modules.append("slow_tracer_modules")
        return admission.ImplicitVerticalAdvectionExecution(
            populate_module_diagnostics=populate_module_diagnostics,
            accumulate_tendency_diagnostics=accumulate_tendency_diagnostics,
            enforce_tendency=enforce_tendency,
            missing_required_slow_modules=tuple(missing_required_slow_modules),
        )

    def set_obc_eta(self, obc_eta: float | torch.Tensor) -> None:
        """Set the prescribed OBC η used at open-boundary cells.

        MITgcm equivalent: OBWeta, OBEeta, OBSeta, OBNeta fields set by
        OBCS_CALC and applied by OBCS_APPLY_ETA (obcs_apply_eta.F).

        Parameters
        ----------
        obc_eta : float or Tensor [ny, nx]
            Prescribed η at open-boundary cells. A scalar applies the
            same value to all boundary cells.  A 2-D tensor allows
            spatially-varying prescribed η (e.g. from HYCOM SSH).
            Default (when never called) is 0.0 for backward compatibility.
        """
        object.__setattr__(self, "_obc_eta", obc_eta)

    def _get_obc_eta(self) -> float | torch.Tensor:
        """Return the current prescribed OBC η (default 0.0)."""
        return 0.0 if self._obc_eta is None else self._obc_eta

    # -- checkpoint contract -------------------------------------------------
    # Restart-critical cross-step state is the closed schema-v2 allowlist in
    # ``defgcm.checkpoint.state``.  Object type and string-key status do
    # not infer ownership: allowlisted unsupported values fail, while all
    # rebuild/current/diagnostic keys are ignored.
    # TendencyInputs re-derives its evaluation-view values from this history
    # at the next step, so the views need not be serialised.
    def export_step_state(self, *, detach: bool = True) -> dict:
        """Export restart-critical cross-step state to a host-resident dict.

        ``detach=True`` (restart) severs the autograd graph and copies to
        CPU; ``detach=False`` (reserved for reverse-mode/adjoint) keeps the
        graph.  No physics behaviour depends on this flag.
        """

        from defgcm.checkpoint.state import encode_step_state_from_stepper

        return encode_step_state_from_stepper(self, detach=detach)

    def import_step_state(self, state: dict, *, device, dtype) -> None:
        """Restore cross-step state into a freshly built stepper.

        Float tensors are coerced to ``dtype`` (model precision); integer /
        bool tensors keep their dtype (masks, counters).
        """

        from defgcm.checkpoint.state import (
            apply_decoded_step_state,
            decode_step_state,
        )

        decoded = decode_step_state(
            state,
            stepper=self,
            device=device,
            dtype=dtype,
        )
        apply_decoded_step_state(self, decoded)

    def _cache_total_phi_hyd_from_diagnostics(self, diagnostics: DiagnosticState) -> None:
        field = diagnostics.fields.get("total_hydrostatic_potential")
        if field is None:
            return
        object.__setattr__(
            self, "_cached_total_phi_hyd", pressure.clone_total_phi_hyd(phi=field.data)
        )

    def _seed_total_phi_hyd_picard(
        self,
        model: SplitStepperModelContext,
        state: PrognosticState,
    ) -> None:
        """MIT ``INI_PRESSURE``-equivalent fixed-point seed for JMD95P.

        MITgcm zeros ``totPhiHyd`` then iterates ``CALC_PHI_HYD`` to converge
        the lagged dynamic pressure before the first dynamic step
        (``model/src/ini_pressure.F:62-176``,
        ``model/src/ini_fields.F:26-33``).  Without the seed, the first
        JMD95P EOS pass would fall back to JMD95Z fixed-depth
        pressure, a ≈1.72 bar omitted dynamic-pressure increment (49-317×
        RMSE degradation by step 1 in the d1a-only ablation); the EOS now
        refuses a missing potential, and the seed's first pass hands it an
        explicit zero.  ``validate_dynamic_pressure_composition`` requires the
        hydrostatic_pressure_gradient module this seed reads.

        Iterates EOS density + HPG diagnostic on ``state``, each time passing
        the previous iterate's ``total_hydrostatic_potential`` to the EOS as
        the evaluation's ``EvaluationInputs.total_phi_hyd``, with MITgcm's
        stopping rule (``pressure.seed_total_phi_hyd_picard``).  A fresh run
        seeds from the state its caller gives, before the initial boundary
        pass and the initial convective adjustment
        (``runner.initialize_fresh_step_state``): INI_PRESSURE runs in
        INI_FIELDS (``initialise_varia.F:225``) before PACKAGES_INIT_VARIABLES
        applies the open-boundary values (``:263``,
        ``packages_init_variables.F:631``, ``obcs_init_variables.F:401``) and
        before CONVECTIVE_ADJUSTMENT_INI (``initialise_varia.F:290``).  An
        initial control already added to that state (casegen's ``temp0``) is
        seen by the seed, as in the adjoint build that runs INI_PRESSURE after
        the controls; MITgcm itself adds them after INI_PRESSURE
        (``packages_init_variables.F:619``), which gives the same values for a
        zero control.

        ``prepare_step_host`` calls this same method whenever a step starts
        with no potential on the stepper: ``release_working_step_state`` clears
        it at every fresh start, segment end and window end, and only the
        runner's fresh initialization, a segment head or a checkpoint that
        saved it restores it.  So a resume from a checkpoint that saved none,
        the direct ``model.step`` API, and a ``model.step`` after a window has
        released its state (``checkpoint/window.py``) seed from the step's
        input state, the last one from a state in the middle of a run.
        MITgcm seeds at fresh starts only (``ini_fields.F:27-33``); a restart
        reads totPhiHyd from the pickup (``read_pickup.F:233-235``).
        """
        # Local imports to avoid cycles
        from defgcm.dynamics.equation_of_state import JMD95EquationOfState, _eos_dump_tap
        from defgcm.dynamics.hydrostatic_pressure import HydrostaticPressureGradient
        from defgcm.solvers.tridiag import forward_mode_requested

        eos = None
        hpg = None
        for module in getattr(model, "dynamics_modules", ()):
            if (
                isinstance(module, JMD95EquationOfState)
                and getattr(module, "dynamic_pressure", False)
            ):
                eos = module
            if isinstance(module, HydrostaticPressureGradient):
                hpg = module
        if eos is None or hpg is None:
            return

        grid = model.grid

        tensors = ()
        if isinstance(state, PrognosticState):
            tensors = (state.temp.data, state.salt.data, state.eta.data, grid.layer_thickness)
            if grid.masks.hFacC is not None:
                tensors += (grid.masks.hFacC,)
        standard_hpg = getattr(hpg.evaluate, "__func__", None) is HydrostaticPressureGradient.evaluate
        # Only the static-geometry reverse path rematerializes this loop.
        # Transforms keep the ordinary Torch program; dynamic hFac and dump
        # writes cannot be replayed from a later grid state.
        rematerialize = (
            torch.is_grad_enabled() and bool(tensors)
            and any(value.requires_grad for value in tensors)
            and not self._nonlin_free_surf_active(grid)
            and not torch._C._are_functorch_transforms_active()
            and not forward_mode_requested(*tensors)
            and not _eos_dump_tap().enabled
            and not ckpt_dump.ENABLED
            and standard_hpg
        )

        new_phi = pressure.seed_total_phi_hyd_picard(
            state=state,
            grid=grid,
            eos=eos,
            hpg=hpg,
            tensors=tensors,
            standard_hpg=standard_hpg,
            rematerialize=rematerialize,
            momentum_pressure_forcing=self.momentum_pressure_forcing,
        )

        if new_phi is not None:
            # Under grad mode the seed keeps the graph of every iteration, so the
            # first step's derivative includes dseed/d(T, S); detached, it left a
            # 2e-5 to 2e-4 relative error in the one-step derivative of the h160
            # fixture.  It requires grad exactly when
            # the potential it came from does, as every later step caches it
            # (otherwise ``step_kernel`` retraces on a requires_grad mismatch of
            # this tensor).
            # Keep the finite-iteration derivative. Checkpoints retain each
            # iterate's inputs and rebuild one EOS/HPG tape at a time, instead
            # of holding all iterations' activations beside the first step.
            object.__setattr__(self, "_cached_total_phi_hyd", new_phi)

    def _sample_external_pnh(self, grid: StructuredCGrid) -> None:
        """Sample the external phi_nh provider at the end-of-step clock. **Eager.**

        The provider's declared contract is ``f(time_seconds: float) ->
        Tensor[nz, ny, nx] | None`` and its only argument is the clock, so the
        whole of its time dependence resolves here.  ``None`` -- the provider
        saying "no field at this time" -- is preserved as ``None``; the step
        body branches on it exactly as it branched on the return value before.

        The cube step body has no phi_nh correction and never called the
        provider, so neither does this: sampling unconditionally would give a
        cube run one external call per step it did not have before, which is a
        visible change to somebody else's function.
        """
        provider = self.external_pnh_provider
        if provider is None or grid.face_count == 6:
            object.__setattr__(self, "_prepared_pnh_phi", None)
            return
        object.__setattr__(
            self, "_prepared_pnh_phi", provider(self._prepared_future_seconds)
        )

    def _apply_pnh_correction(
        self,
        u_new: torch.Tensor,
        v_new: torch.Tensor,
        grid,
        surface_mask: Optional[torch.Tensor],
        dt_mom: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the external nonHydrostatic phi_nh correction.

        The provider is a function of the clock alone -- it is handed the
        end-of-step time and returns a field or ``None`` -- so prepare samples
        it (``_sample_external_pnh``) and the step body consumes the tensor.
        Sampling in prepare supplies the host float required by the provider
        while keeping the traced step dependent only on the prepared tensor.
        """
        if self.external_pnh_provider is None:
            return u_new, v_new
        # A bare attribute read, like the two prepared clocks beside it: if
        # prepare did not run there is no sample, and that must raise rather
        # than quietly drop the correction (the explicit-input requirement).
        phi = self._prepared_pnh_phi
        if phi is None:
            return u_new, v_new
        # correction_step.F:157: nhFac = pfFacMom * implicitNHPress.
        # The supported internal NH path fixes implicitNHPress at one.
        if not self.momentum_pressure_forcing:
            return u_new, v_new
        return pressure.apply_pnh_correction_from_phi(
            u_new=u_new,
            v_new=v_new,
            grid=grid,
            dt_mom=dt_mom,
            phi=phi,
            surface_mask=surface_mask,
        )

    # ------------------------------------------------------------------
    # Nonlinear free surface: surface-thickness time layers
    # ------------------------------------------------------------------

    def _uses_freshwater_history(self) -> bool:
        """Whether the step keeps MITgcm's PmEpR across steps (``_pmep_r``).

        Only a nonlinear free surface with real freshwater flux reads it, in
        the surface salt forcing (``external_forcing_surf.F:261-288``).
        """
        return bool(self.nonlin_free_surf >= 1 and self.use_real_freshwater_flux)

    def _pmep_r_for_step(self, grid, dtype, device, *, first_step: bool,
                         empmr_mass_flux: torch.Tensor | None) -> torch.Tensor | None:
        """PmEpR [kg m-2 s-1] the surface forcing of this step reads, or None.

        Its time level follows staggerTimeStep (``external_forcing_surf.F:
        158-166``), and both values are ``-EmPmR*maskInC``:

        * synchronous: the previous continuity's, so that PmEpR lags EmPmR by
          one step and matches the volume change d(etaH)/dt the tracers of this
          step see.  A fresh run's first step reads zero: INITIALISE_VARIA's
          INTEGR_CONTINUITY is the only call with ``myIter == nIter0`` and sets
          it (``integr_continuity.F:163-171``, ``ini_nlfs_vars.F:59``).
          ``first_step`` is the step-entry test of a fresh run (no etaH yet); a
          continued step must find the key.
        * staggered: this step's, from ``empmr_mass_flux``.  EXTERNAL_FORCING_SURF
          sets it before the surface forcing, and the tracers stepped after this
          step's continuity see that continuity's freshwater.  The carried
          ``_pmep_r`` is not read, on a first step or after a restart:
          synchronous stepping needs the PmEpR history, staggered stepping
          overwrites it within the step.
        """
        if not self._uses_freshwater_history():
            return None
        if not self.exact_conserv:
            raise RuntimeError(
                "nonlin_free_surf>=1 with use_real_freshwater_flux needs exact_conserv: "
                "MITgcm sets PmEpR in the exactConserv continuity (integr_continuity.F:"
                "90-200) and refuses a nonlinear free surface without it (config_check.F)")
        if getattr(self, "_resolved_volume_freshwater", None) is not True:
            raise RuntimeError(
                "nonlin_free_surf>=1 with use_real_freshwater_flux: the surface forcing "
                "branch was not resolved at model assembly (resolve_volume_freshwater)")
        if self.stagger_tracer_stepping:
            return ifs_freshwater.continuity_freshwater_history(empmr_mass_flux, grid=grid)
        pmep = self._pmep_r
        if pmep is not None:
            return pmep
        if not first_step:
            raise RuntimeError(
                "a continued nonlinear-free-surface run with real freshwater flux has no "
                "PmEpR history (_pmep_r); its restart state predates the key")
        return torch.zeros((grid.ny, grid.nx), dtype=dtype, device=device)

    def _relaxation_surface_hfac_c(self, grid) -> torch.Tensor | None:
        """Surface-level hFacC [ny, nx] of the restoring flux KPP reads, or None.

        MITgcm's FORCING_SURF_RELAX carries the surface thickness of the tracer
        stage that applies the flux: it generates it with the step-start
        ``_hFacC`` and, under staggerTimeStep with a nonlinear free surface,
        multiplies by ``hFac_surfC/_hFacC`` because the staggered tracers run
        after UPDATE_SURF_DR (``forcing_surf_relax.F:95-104,113-151``).  KPP
        reads that surfaceForcingT before the geometry update
        (``do_oceanic_phys.F:579,956``), when the grid still holds the
        step-start layer and ``_surface_geometry.surface`` the next one.
        ``None`` only for a grid without hFacC.
        """
        if grid.masks.hFacC is None:
            return None
        if self.stagger_tracer_stepping and self._surface_dr_active(grid):
            return self._surface_geometry.surface[0].select(dim=-3, index=0)
        return grid.masks.hFacC.select(dim=-3, index=0)

    def _nonlin_free_surf_active(self, grid: StructuredCGrid) -> bool:
        """True when this run takes MITgcm's ``nonlinFreeSurf > 0`` geometry path."""
        return self.nonlin_free_surf >= 1 and ifs_geometry.surface_geometry_has_masks(grid.masks)

    def _surface_dr_active(self, grid: StructuredCGrid) -> bool:
        """The ``select_rStar = 0`` branch: only the surface level moves (CALC/UPDATE_SURF_DR)."""
        return self._nonlin_free_surf_active(grid) and self.select_r_star == 0

    def _r_star_active(self, grid: StructuredCGrid) -> bool:
        """The ``select_rStar > 0`` branch: every level moves (CALC/UPDATE_R_STAR)."""
        return self._nonlin_free_surf_active(grid) and self.select_r_star == 2

    def _initialise_r_star(self, model, grid, state, *, dt_freesurf) -> None:
        """INITIALISE_VARIA's r* calls (``ifs_geometry.initialise_r_star``).

        The columns are built once from the resting geometry the grid holds
        before the first install.  ``rStarAreaWeight`` is off when the assembled
        momentum form is vector-invariant with ``selectKEscheme`` 1 or 3
        (``calc_r_star.F:64-68``), momentum stepping or not;
        ``prepare_step_host`` has refused a vector-invariant form whose KE
        choice no module carries.
        """
        from defgcm.dynamics.vector_invariant_momentum import VectorInvariantMomentumAdvection

        columns = self._r_star.columns
        if columns is None:
            eta = state.eta.data
            ke_schemes = [module.select_ke_scheme for module in model.dynamics_modules
                          if isinstance(module, VectorInvariantMomentumAdvection)]
            columns = ifs_geometry.r_star_columns(
                grid, (grid.masks.hFacC.clone(), grid.masks.hFacW.clone(), grid.masks.hFacS.clone()),
                area_weight=not (self._resolved_vector_invariant_momentum and ke_schemes[0] in (1, 3)),
                dtype=eta.dtype, device=eta.device,
            )
        eta_seed = self._continuity.eta_h
        if eta_seed is None:
            eta_seed = state.eta.data
        object.__setattr__(self, "_r_star", ifs_geometry.initialise_r_star(
            grid, columns, eta_seed, dt_free_surf=dt_freesurf,
        ))

    def _calc_r_star(self, grid, eta_h, *, dt_freesurf) -> None:
        geometry = self._r_star
        object.__setattr__(self, "_r_star", ifs_geometry.RStarGeometry(
            geometry.columns,
            ifs_geometry.calc_r_star_factors(
                grid, geometry.columns, eta_h, previous=geometry.factors,
                dt_free_surf=dt_freesurf,
            ),
        ))

    def _check_nonlin_free_surf_preconditions(self, grid: StructuredCGrid) -> None:
        """Reject grids where MITgcm's ``Rmin_surf`` floor does not collapse.

        ``compute_surface_hfac`` applies ``hFac >= hFacInf`` where MITgcm clamps
        the r-position by ``Rmin_surf`` (``ini_nlfs_vars.F:156-171``).  The two
        agree when ``Ro_surf == 0`` -- structural here, LOCOMgcm has neither
        p-coordinates nor an ice shelf -- **and** no wet column is shallower than
        the top layer, because a shallower wet neighbour raises ``Rmin_tmp``
        above ``rF(2)``.  Fail loudly rather than run a floor that silently
        differs from the reference model.
        """
        # R_low is resting geometry. Restart installs the lagged hFac layer
        # before prepare; a negative eta must not turn a valid static column
        # into a shallow one (ini_nlfs_vars.F:158-170).
        resting = self._surface_geometry.resting
        hFacC = grid.masks.hFacC if resting is None else resting[0]
        drF = grid.layer_thickness.to(dtype=hFacC.dtype, device=hFacC.device)
        column_depth = (hFacC * drF.view(-1, 1, 1)).sum(dim=0)
        wet = hFacC[0] > 0
        shallow = wet & (column_depth < float(drF[0]))
        if bool(shallow.any()):
            n = int(shallow.sum())
            raise ValueError(
                f"nonlin_free_surf={self.nonlin_free_surf} on a grid with {n} wet "
                f"column(s) shallower than the top layer ({float(drF[0])} m): "
                "MITgcm's Rmin_surf floor (ini_nlfs_vars.F:156-171) then depends "
                "on the neighbours' R_low and no longer collapses to hFacInf, "
                "which is the only form compute_surface_hfac implements"
            )

    def _get_coefficients(
        self,
        grid: StructuredCGrid,
        dt_mom: float,
        dt_freesurf: float,
        dtype: torch.dtype,
        device: torch.device,
        ctx=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get or compute Helmholtz operator coefficients (cached by (dt_mom, dt_freesurf, ...)).

        Returns the edge-coefficient triple ``(aW_edge, aS_edge, aC)``.

        When *ctx* is provided (multi-rank), the cache key is augmented with the
        decomposition and rank so each ownership layout stores its coefficients independently
        (different ranks own different face subsets → different local tensors).
        The coefficients are assembled via ``_assemble_helmholtz_coefficients``
        which uses the distributed exchange to reconcile cross-rank seam slots.
        """
        alpha_beta = self.implicit_surf_press * self.implicit_div_2d_flow
        interior_mask = None
        if grid.boundary_topology.open_x or grid.boundary_topology.open_y:
            maskInC, _, _ = _compute_interior_mask(grid, dtype, device)
            interior_mask = maskInC

        # nonlinFreeSurf > 2: hFac changes every step, must recompute
        if self.nonlin_free_surf > 2:
            return _assemble_helmholtz_coefficients(
                grid, self.gravity, dt_mom, dt_freesurf, dtype, device, alpha_beta,
                interior_mask=interior_mask,
                ctx=ctx,
            )

        # Equal rank numbers can own different faces in different decompositions.
        layout_key = (ctx.decomposition, ctx.rank) if ctx is not None else None
        return coefficients.memoized_helmholtz_coefficients(
            grid=grid, gravity=self.gravity, dt_mom=dt_mom,
            dt_freesurf=dt_freesurf, dtype=dtype, device=device,
            alpha_beta=alpha_beta, interior_mask=interior_mask,
            layout_key=layout_key, ctx=ctx,
        )

    @torch._dynamo.disable
    def _get_cg3d_coefficients(
        self,
        grid: StructuredCGrid,
        time_scales: TimeStepBundle,
        dtype: torch.dtype,
        device: torch.device,
        mask_in_c: torch.Tensor | None = None,
    ):
        """Get cached linFS CG3D operator/preconditioner coefficients."""
        return coefficients.cg3d_coefficients(
            grid=grid, params=self.nonhydrostatic, gravity=self.gravity,
            time_scales=time_scales, dtype=dtype, device=device, mask_in_c=mask_in_c,
        )

    def _prepare_cg3d_coefficients(self, grid, state, bundle) -> None:
        """Publish the fixed-geometry CG3D coefficients, so the step only reads them.

        ``_get_cg3d_coefficients`` is ``@torch._dynamo.disable`` because its cold
        build runs the geometry exchanges under ``constant_transport()``, which
        Dynamo does not inline.  The traced step called it every step, so a warm
        memo still broke the graph there and ``DEFGCM_COMPILE_FULLGRAPH=1`` refused
        the nonhydrostatic step.  With fixed geometry the pair depends only on the
        memo key (grid, the two time steps, dtype, device), so the identical call
        is made here on the key the step body presents.  Active geometry
        (time-varying hFac, or geometry that requires grad) publishes nothing and
        the step builds the pair on its own geometry, as before.

        A nonlinear free surface publishes nothing either: the body rewrites
        hFac (``_apply_surf_dr``) before the CG3D solve, and that is also where
        ``hfac_time_varying`` is raised, so on the first step of a run or of a
        restart the flag still reads False here while the solve needs the
        rewritten geometry.
        """
        prepared = None
        if (
            self.nonhydrostatic.nonhydrostatic
            and self.external_pnh_provider is None
            and not self._nonlin_free_surf_active(grid)
        ):
            temp = state.temp.data
            mask_in_c = None
            if grid.boundary_topology.open_x or grid.boundary_topology.open_y:
                mask_in_c, _, _ = _compute_interior_mask(grid, temp.dtype, temp.device)
            if coefficients.cg3d_geometry_fixed(
                grid=grid, dtype=temp.dtype, device=temp.device,
                mask_in_c=mask_in_c, gravity=self.gravity,
            ):
                prepared = self._get_cg3d_coefficients(
                    grid, bundle, temp.dtype, temp.device, mask_in_c=mask_in_c
                )
        object.__setattr__(self, "_prepared_cg3d_coefficients", prepared)

    # -- KPP vertical mixing (regional path) ----------------------------------
    @torch._dynamo.disable
    def prepare_step_runtime(self, model, grid, state) -> None:
        """Run top-of-step host preparation behind one disabled boundary.

        One wrapper is required so Dynamo does not resume and break
        again between grid residency and KPP preparation.
        Traced KPP static construction can split tensor
        constructors across CPU and CUDA.
        """
        ensure_grid_device_resident(grid, state.u.data.device)
        if getattr(self, "kpp_enabled", False):
            self.prepare_kpp_static(
                model, grid, state.temp.data.dtype, state.temp.data.device
            )

    @torch._dynamo.disable
    def prepare_kpp_static(self, model, grid, dtype, device) -> "KppStatic":
        """Build and cache regional KPP static tensors outside traced code."""
        inputs = self._kpp_static_inputs
        if inputs.static is not None:
            return inputs.static

        modules = getattr(model, "dynamics_modules", ()) or ()
        layer_thickness = grid.layer_thickness.to(dtype=dtype, device=device)
        coriolis = resolve_kpp_coriolis(
            modules=modules, profile_c=self.kpp_coriolis_profile_c,
            metrics=grid.metrics, ny=grid.ny, nx=grid.nx, dtype=dtype, device=device,
            omega=omega_from_rotation_period(self.rotation_period),
        )

        inputs = prepare_kpp_static(
            grid, layer_thickness, coriolis,
            coriolis_override=self.kpp_coriolis,
            rho_const=inputs.rho_const, dtype=dtype, device=device,
            omega=omega_from_rotation_period(self.rotation_period),
        )
        object.__setattr__(self, "_kpp_static_inputs", inputs)
        return inputs.static

    def _prepare_regional_kpp_inputs(self, state, model, grid, dtype, device, *, forcing, freezing_flux):
        """Resolve explicit KPP inputs at the caller's scientific time station.

        Static preparation precedes this adapter. Surface forcing comes from
        the current producer bundle; the EOS pressure is the stepper EOS
        module's own pressure field with the hydrostatic history at this call.
        The numerical consumer does not read the model or cache.
        """
        rho0 = float(self._kpp_static_inputs.rho_const)
        static = self._kpp_static_inputs.static
        sfT, sfS, sfU, sfV, qsw = prepare_regional_kpp_surface_inputs(
            static=static,
            surf_forc_t=forcing.surf_forc_t,
            surf_forc_u=forcing.surf_forc_u,
            surf_forc_v=forcing.surf_forc_v,
            surf_forc_s=forcing.surf_forc_s,
            surf_qsw=forcing.surf_qsw,
            allow_freezing=self.allow_freezing, freezing_flux=freezing_flux,
            dtype=dtype, device=device,
        )
        eos_module = kpp_eos_module(model)
        eos_type = kpp_eos_type(kpp_eos_name(eos_module))
        # MITgcm's STATEKPP evaluates FIND_RHO_2D at PRESSURE_FOR_EOS
        # (kpp_routines.F:1849-1910), the pressure the model EOS uses.  A
        # dynamic-pressure EOS (JMD95P, and MDJWF, which MITgcm always runs
        # with selectP_inEOS_Zc=2) therefore hands KPP this step's EOS
        # pressure array, rhoConst*(totPhiHyd + phiRef(2k))
        # (pressure_for_eos.F:71-86); self._cached_total_phi_hyd is the
        # array this step's EOS reads (step 0: the INI_PRESSURE Picard seed).
        # Only JMD95Z, whose pressure is the depth profile, keeps kpp.py's
        # rho0*g*z. Before this, JMD95P fell back to that profile too.
        eos_pressure_pa = None
        uses_pressure = eos_type == "mdjwf" or getattr(eos_module, "dynamic_pressure", False)
        if uses_pressure:
            eos_pressure_pa = self._eos_pressure_at_call(eos_module, grid, target=state.temp.data)
        return prepare_regional_kpp_inputs(
            static=static, rho_const=rho0,
            surf_forc_t=sfT, surf_forc_s=sfS, surf_forc_u=sfU, surf_forc_v=sfV,
            surf_qsw=qsw, eos_pressure_pa_at_call=eos_pressure_pa,
            eos_module=eos_module, run_values=self._kpp_run_values,
            forcing_penetrating_sw=forcing.penetrating_sw,
            tracer_vertical_diffusivity=self.tracer_vertical_diffusivity_kr,
            convective_implicit=self.kpp_convective_implicit,
        )

    def _eos_pressure_at_call(self, eos_module, grid, *, target):
        """PRESSURE_FOR_EOS of this step (pressure_for_eos.F:71-86) for the column schemes."""
        pressure_field = getattr(eos_module, "_pressure_field", None)
        if pressure_field is None:
            raise ValueError(
                f"vertical mixing requires the pressure-field provider of {eos_module.name}"
            )
        if (
            getattr(eos_module, "dynamic_pressure", False)
            and self._cached_total_phi_hyd is None
        ):
            raise ValueError(
                "vertical mixing with a dynamic-pressure EOS requires lagged total_phi_hyd"
            )
        return pressure_field(
            grid, dtype=target.dtype, device=target.device, target=target,
            prev_total_phi_hyd=(self._cached_total_phi_hyd
                                if getattr(eos_module, "dynamic_pressure", False) else None),
        ).expand_as(target)

    def _mixing_densities(self, bounded, model, grid):
        """Step-start ``(rhoInSitu, rhoKm1)`` of the vertical mixing schemes (columns.mixing_densities)."""
        eos_module = columns.mixing_equation_of_state(model)
        temp = bounded.temp.data
        pressure = None
        if eos_module.name != "linear_equation_of_state":
            pressure = self._eos_pressure_at_call(eos_module, grid, target=temp)
        return columns.mixing_densities(eos_module, temp=temp, salt=bounded.salt.data, pressure=pressure)

    def _kpp_nonlocal_diffusivities(self, kpp_fields, gm_contributions):
        """``(T, S)`` diffusivities of KPP's nonlocal flux: KPPdiffKzT/S, plus
        GM/Redi Kwz when ``kpp_ghat_use_total_diffus`` and a gmredi module are both
        on (``tmpFac``, ``kpp_transport_t.F:69-82``, ``kpp_transport_s.F:86-95``)."""
        if self.kpp_ghat_use_total_diffus and gm_contributions:
            kwz = gm_contributions[0].kwz
            return kpp_fields.diff_kzt + kwz, kpp_fields.diff_kzs + kwz
        return kpp_fields.diff_kzt, kpp_fields.diff_kzs

    def _gm_residual_tracer_w(self, diagnostics, tracer_w):
        """MITgcm ``wFld`` for GAD_IMPLICIT_R and the continuity term: ``wVel`` plus the GM
        bolus ``w`` under GM_AdvForm (``thermodynamics.F:258-276``, both MIT upward-positive);
        the momentum keeps ``wVel``."""
        gmredi = self._package_modules.gmredi
        if tracer_w is None or gmredi is None or not gmredi.adv_form:
            return tracer_w
        return TracerWNpHalfMIT(data=tracer_w.data + diagnostics.tensors["gm_bolus_w"])

    def _gm_kwz_contributions(self, diagnostics, grid, *, dtype, device):
        """GMREDI_CALC_DIFF's Kwz for CALC_3D_DIFFUSIVITY when a GM/Redi module is configured."""
        if self._package_modules.gmredi is None:
            return ()
        mask_in_c, _, _ = _compute_interior_mask(grid, dtype, device)
        return (columns.KwzContribution(kwz=diagnostics.tensors["gm_kwz"], mask_in_c=mask_in_c),)

    @property
    def reads_kpp_forcing(self) -> bool:
        """Whether a step reads the surface forcing record: KPP, and GGL90 for
        surfaceForcingU/V (do_oceanic_phys.F:579, ggl90_calc.F:766-797)."""
        return self.kpp_enabled or self.vertical_mixing_scheme == "ggl90"

    def _kpp_forcing_producers(self):
        """Yield selected sources in wind-then-tracer harvest order."""
        wind = self._package_modules.kpp_wind
        if wind is not None:
            tau_x = getattr(wind, "_current_tau_x", None)
            tau_y = getattr(wind, "_current_tau_y", None)
            if (tau_x is not None and tau_y is not None) or getattr(wind, "use_bulk_stress", False):
                yield "era5_wind_forcing", wind, tau_x, tau_y
        tracer = self._package_modules.kpp_tracer
        if tracer is not None:
            yield getattr(tracer, "name", ""), tracer, None, None

    def _populate_kpp_forcing_from_model(self, model, state, grid, dtype, device, *,
                                         forcing_inputs: KppStepForcing,
                                         pmep_r: torch.Tensor | None = None) -> KppStepForcing:
        """Collect this step's KPP forcing from the current producer samples.

        Only present producers replace their fields in the returned local copy.
        Returned fields stay local to the regional step and its tracer station.
        SurfaceForcing signs and units are retained from each producer.
        ``_kpp_static_inputs.rho_const`` is aligned to the forcing modules' reference density so
        the KPP buoyancy uses the same rhoConst as the surface forcing.
        """
        from defgcm.dynamics.era5_wind_forcing import (
            ERA5WindForcing,
            _surface_face_masks,
        )

        rho0 = None
        for nm, m, tau_x, tau_y in self._kpp_forcing_producers():
            if nm == "era5_wind_forcing":
                rho0 = float(m.reference_density)
                if isinstance(m, ERA5WindForcing):
                    tau_x_on_u, tau_y_on_v = m._ensure_face_stress(
                        grid, dtype=dtype, device=device, state=state
                    )
                else:
                    mask_w, mask_s = _surface_face_masks(
                        grid, dtype=dtype, device=device
                    )
                    tau_x_on_u, tau_y_on_v = prepare_kpp_face_stress(
                        tau_x, tau_y, grid, face_mask_w=mask_w, face_mask_s=mask_s,
                        dtype=dtype, device=device,
                    )
                    setattr(m, "_current_tau_x_on_u", tau_x_on_u)
                    setattr(m, "_current_tau_y_on_v", tau_y_on_v)
                sfU, sfV = prepare_kpp_wind_forcing(
                    tau_x_on_u, tau_y_on_v, self._prepared_mass_to_r_unit,
                    ny=grid.ny, nx=grid.nx,
                )
                forcing_inputs = dataclasses.replace(
                    forcing_inputs, surf_forc_u=sfU, surf_forc_v=sfV,
                )
            elif nm == "era5_bulk_forcing":
                # Produce the SurfaceForcing bundle NOW (before the tendency
                # evaluation) so KPP gets THIS step's fields.  era5_bulk only
                # sets _current_forcing inside compute/evaluate, which run AFTER
                # this diagnose — so at step 0 it would still be None and KPP
                # would reject the unset surf_forc_t. The harvest READS the bundle;
                # it never re-derives sfT itself (single produce point).
                if state is not None and hasattr(m, "compute_surface_fluxes"):
                    m.compute_surface_fluxes(state)
                forcing = getattr(m, "_current_forcing", None)
                if forcing is None:
                    break
                sfT, sfS, qsw = prepare_kpp_surface_forcing(
                    forcing.sfT, forcing.sfS, forcing.qsw,
                    ny=grid.ny, nx=grid.nx, dtype=dtype, device=device,
                )
                forcing_inputs = dataclasses.replace(
                    forcing_inputs, surf_forc_t=sfT,
                    surf_qsw=qsw, penetrating_sw=forcing.penetrating_sw,
                    surf_forc_s=sfS,
                )
                if rho0 is None:
                    rho0 = float(forcing.rho_const)
            elif nm == "wpflux_surface_forcing":
                # The module's producer, with the step's PmEpR: KPP_FORCING_SURF
                # reads the same surfaceForcingS as APPLY_FORCING_S.
                forcing = m.surface_forcing(state, pmep_r=pmep_r)
                sfT, sfS, qsw = prepare_kpp_surface_forcing(
                    forcing.sfT, forcing.sfS, forcing.qsw,
                    ny=grid.ny, nx=grid.nx, dtype=dtype, device=device,
                )
                forcing_inputs = dataclasses.replace(
                    forcing_inputs, surf_forc_t=sfT,
                    surf_qsw=qsw, penetrating_sw=forcing.penetrating_sw,
                    surf_forc_s=sfS,
                )
                if rho0 is None:
                    rho0 = float(forcing.rho_const)
            elif nm == "surface_heat_salt_flux":
                ref = float(m.reference_density)
                # KPP_FORCING_SURF consumes the same surfaceForcingT as
                # APPLY_FORCING_T, including FORCING_SURF_RELAX.
                sfT = m.surface_heat_forcing(
                    state, grid, surface_hfac_c=self._relaxation_surface_hfac_c(grid),
                )
                sfT, sfS = prepare_kpp_heat_salt_forcing(
                    sfT, m.salt_flux, ny=grid.ny, nx=grid.nx, dtype=dtype, device=device,
                )
                forcing_inputs = dataclasses.replace(
                    forcing_inputs, penetrating_sw=False,
                    surf_forc_t=sfT, surf_forc_s=sfS,
                )
                if rho0 is None:
                    rho0 = ref

        relaxation = self._package_modules.surface_relaxation
        if relaxation is not None:
            # KPP_FORCING_SURF reads surfaceForcingT/S, which FORCING_SURF_RELAX
            # starts before the surface fluxes add to them
            # (external_forcing_surf.F:173-234).
            # no ice here: KPP beside the seaice module is refused at assembly
            relax = relaxation.surface_forcing(state, grid, ice_area=None)
            forcing_inputs = dataclasses.replace(
                forcing_inputs,
                surf_forc_t=relax.sfT.reshape(1, grid.ny, grid.nx) + forcing_inputs.surf_forc_t,
                surf_forc_s=relax.sfS.reshape(1, grid.ny, grid.nx) + forcing_inputs.surf_forc_s,
            )

        if rho0 is not None:
            object.__setattr__(self, "_kpp_static_inputs", dataclasses.replace(
                self._kpp_static_inputs, rho_const=rho0,
            ))
        if forcing_inputs.surf_forc_u is None and self._package_modules.surface_stress is None:
            # No module forces momentum at the surface: MITgcm's fu/fv stay at
            # their zero initial value (ini_ffields.F:45-46), so
            # surfaceForcingU/V = fu*mass2rUnit = 0 (external_forcing_surf.F:222-224).
            forcing_inputs = dataclasses.replace(
                forcing_inputs,
                surf_forc_u=torch.zeros((1, grid.ny, grid.nx + 1), dtype=dtype, device=device),
                surf_forc_v=torch.zeros((1, grid.ny + 1, grid.nx), dtype=dtype, device=device),
            )

        return forcing_inputs

    # -- cubed-sphere wind-stress hand-off ------------------------------------
    def set_cube_wind_stress(
        self,
        tau_x: torch.Tensor | None,
        tau_y: torch.Tensor | None,
    ) -> None:
        """Register surface wind-stress fields for the cube step.

        ``tau_x`` / ``tau_y`` are ``[6, ny, nx]`` compact C-grid face stress
        in ``N m-2``, positive along the face-local x/y axes.
        The cube step places them directly at the west/
        south C-grid face slots without center-to-face averaging and applies
        ``gU += tau / (rhoConst * drF[0] * hFacW[0])`` to the surface layer
        (MITgcm ``EXTERNAL_FORCING_U``).  Their AB2 placement is selected by
        ``cube_assembly_policy.mom_forcing_out_ab``.  Pass ``None`` to disable.
        """
        object.__setattr__(
            self, "_cube_surface_forcing",
            dataclasses.replace(self._cube_surface_forcing, tau_x=tau_x, tau_y=tau_y),
        )

    def set_cube_surface_fluxes(
        self,
        qnet: torch.Tensor | "SurfaceForcing" | None,
        qsw: torch.Tensor | None = None,
        empmr: torch.Tensor | None = None,
        *,
        rho_const_fresh: float | None = None,
        penetrating_sw: bool | None = None,
    ) -> None:
        """Register the surface heat/shortwave/freshwater fields for the cube step.

        All fields are ``[6, ny, nx]`` compact cube-face surface fields in **MITgcm
        convention** (the producer/ingest converts at the data boundary, docs/NUMERICS.md):
        - ``qnet``  net heat flux ``Qnet`` [W/m²], upward-positive (``>0`` cools);
        - ``qsw`` net shortwave ``Qsw`` [W/m²], upward-positive. The effective
          ``penetrating_sw`` mode, not the field's value or presence, controls
          heat splitting and penetration. Raw callers use the stepper setting
          unless they explicitly override it here;
        - ``empmr`` net freshwater flux EmPmR as the **bulk volume flux** [m/s]
          (= exf flux); the cube step multiplies by ``rhoConstFresh`` to the MITgcm
          mass flux [kg/m²/s] before the virtual salt flux; ``None`` ⟹ no salt forcing.

       The production path passes a :class:`SurfaceForcing` as ``qnet``.  Its
        already-produced ``sfT``/``sfS`` fields are the single source read by cube
        KPP, while its raw fields are held in the cube current-step record.
       The tensor form remains for compatibility.  Pass ``qnet=None`` to
        clear the heat input.
        """
        from defgcm.dynamics.surface_flux import SurfaceForcing

        if isinstance(qnet, SurfaceForcing):
            if qsw is not None or empmr is not None or rho_const_fresh is not None or penetrating_sw is not None:
                raise ValueError(
                    "SurfaceForcing bundle is mutually exclusive with raw cube flux arguments"
                )
            if qnet.qnet is None or qnet.empmr is None:
                raise ValueError(
                    "cube SurfaceForcing bundle requires raw qnet and empmr fields"
                )
            forcing = qnet
            object.__setattr__(
                self, "_cube_surface_forcing",
                dataclasses.replace(
                    self._cube_surface_forcing,
                    qnet=forcing.qnet,
                    qsw=forcing.qsw,
                    penetrating_sw=forcing.penetrating_sw,
                    empmr=forcing.empmr,
                    rho_const_fresh=float(forcing.rho_const_fresh),
                    bulk=forcing,
                ),
            )
            return

        object.__setattr__(
            self, "_cube_surface_forcing",
            dataclasses.replace(
                self._cube_surface_forcing,
                qnet=qnet,
                qsw=qsw,
                penetrating_sw=(
                    self.penetrating_sw if penetrating_sw is None else penetrating_sw
                ),
                empmr=empmr,
                rho_const_fresh=(
                    self._cube_surface_forcing.rho_const_fresh
                    if rho_const_fresh is None else float(rho_const_fresh)
                ),
                bulk=None,
            ),
        )

    def _step_cube(
        self,
        state: PrognosticState,
        model: SplitStepperModelContext,
        bundle: TimeStepBundle,
        *,
        time_seconds: float = 0.0,
    ) -> tuple[PrognosticState, torch.Tensor, torch.Tensor]:
        return _step_cube_impl(
            self,
            state,
            model,
            bundle,
            time_seconds=time_seconds,
        )

    def _step_index_value(self) -> int:
        """Exact int of the completed-step counter (a 0-d int64 tensor).

        A Python int is EQUALS_MATCH-guarded when a compiled step reads it, so
        the counter -- advanced once per step -- recompiled the step frame every
        step until the limit sent it eager.  A tensor is guarded on dtype and
        shape only.  Eager callers read it through this helper; a plain int
        (fresh stepper, restored checkpoint) is accepted as before.
        """
        from defgcm.checkpoint.state import require_exact_int

        value = 0 if self._step_index is None else self._step_index
        if isinstance(value, torch.Tensor):
            return int(value.item())
        return require_exact_int(value, path="stepper._step_index")

    def _validate_step_index_state(self) -> int:
        """Host validation of the step counter. **Eager.**

        Kept out of ``_step_impl``; the frame it raises from
        is the eager ``prepare_step_host``.  Returns the validated counter.
        The momentum solve reads the lagged w the tendencies read
        (``_evaluation_inputs``), one stored tensor with no step stamp of its
        own, so there is no second record here to compare the counter with.
        """
        if self._step_index is None:
            raise ValueError(
                "implicit vertical advection step state is not initialized: "
                "missing _step_index"
            )
        return self._step_index_value()

    def advance_step_index(self) -> None:
        """Advance the completed-step counter. **Eager.**

        Counter advancement belongs to defgcm/step/finalize.py, outside the
        traced region. Reading a changing host counter in the compiled step
        would specialize its frame on each value.

        The value is "the cached counter plus one", the counter proved an exact
        int in ``prepare_step_host``.
        """
        # Stored as a 0-d int64 tensor, not an int: see ``_step_index_value``.
        object.__setattr__(self, "_step_index", torch.tensor(
            self._step_index_value() + 1, dtype=torch.int64, device="cpu"))

    def persist_step_outputs(
        self,
        model: SplitStepperModelContext,
        state: PrognosticState,
    ) -> PrognosticState:
        """Publish retained outputs under the compile binding's ownership contract.

        Ordinary outputs transfer directly; graph-pool and inference outputs
        acquire ordinary storage outside the compiled region.

        A finalize step (``defgcm/step/finalize.py``); the
        ``@torch._dynamo.disable`` it carried while the compiled region ended
        after it is no longer needed.

        Regional and cube histories acquire ordinary storage with the state.
        """
        from defgcm.step.output_ownership import publish_value, shared_storages

        retained = {}
        for name in self.OWNED_STEP_FIELDS:
            if name in ("_step_index", "_kpp_static_inputs", "_previous_eta"):
                continue
            value = getattr(self, name)
            if value is not None:
                retained[name] = value
        phi = (None if self._cached_total_phi_hyd is None
               else _carry(self._cached_total_phi_hyd))
        # One storage per published name (``publish_value``'s ``shared``).
        shared = shared_storages(retained, phi, state)
        for key, value in retained.items():
            object.__setattr__(self, key, publish_value(model, value, shared))
        if phi is not None:
            object.__setattr__(self, "_cached_total_phi_hyd", publish_value(model, phi, shared))
        return publish_value(model, state, shared)

    def step_kernel(
        self,
        state: PrognosticState,
        model: SplitStepperModelContext,
        plan: "StepPlan",
        *,
        ab_history: dict[str, torch.Tensor] | None = None,
        hydro_history=None,
        boundary_evaluation=None,
    ) -> StepOutcome:
        """The traced phase of one step: tensors in, tensors out.

        Every host decision this used to make on the way in is already in
        ``plan``, and everything it used to do on the way out is finalize's.
        The state and solver diagnostics leave together in ``StepOutcome``.

        The clock enters as ``plan.times.current``, a 0-d tensor, not as the
        host float: a per-step Python float in the traced signature is
        specialised (``kwargs['time_seconds'] == 975.0`` failed on every step
        of the static-shape fullgraph arm).
        On the prepared posture the step body never consumes the value -- the
        two clocks it needs were sampled eagerly and are installed from the
        prepared payload -- so the tensor reaches only the legacy branches that
        posture does not take.  ``plan.times.bundle`` is run-constant and stays
        a host object.

        ``plan.boundary`` is the second half of the same rule: the two clocks'
        pre-sampled device payload arrives on this signature instead of being
        fetched from the model inside the traced body.
        """
        if ab_history is not None:
            return self._step_impl(
                state, model, plan,
                ab_history=ab_history,
                hydro_history=hydro_history,
                boundary_evaluation=boundary_evaluation,
            )
        return self._step_impl(
            state,
            model,
            plan,
            boundary_evaluation=boundary_evaluation,
        )

    def step(
        self,
        state: PrognosticState,
        model: SplitStepperModelContext,
        dt: float | TimeStepBundle,
        *,
        time_seconds: float = 0.0,
    ) -> PrognosticState:
        """Advance one step and return its published state. **Eager.**"""
        return self.step_outcome(state, model, dt, time_seconds=time_seconds).state

    def step_outcome(
        self,
        state: PrognosticState,
        model: SplitStepperModelContext,
        dt: float | TimeStepBundle,
        *,
        time_seconds: float = 0.0,
    ) -> StepOutcome:
        """Run all three phases of one step. **Eager.**

        The direct-API entry point: prepare, the step body, then the step
        boundary.  The runner drives the same three phases itself so it can
        compile the middle one alone (``defgcm/step/``); this method is the
        reference implementation both postures agree with, and it no longer
        needs a ``torch._dynamo.is_compiling()`` test to know which one it is
        running under, because the traced region can no longer reach it.
        """
        bundle = coerce_time_step_bundle(dt)
        boundary, kpp_forcing_inputs = self.prepare_step_host(
            model, state, bundle, time_seconds=time_seconds
        )
        from defgcm.step.contract import ShapeContract, StepPlan, StepTimes
        from defgcm.step.plan import _step_index_tensor

        plan = StepPlan(
            gates=self.gates, times=StepTimes.sample(current_seconds=time_seconds, bundle=bundle),
            step_index=_step_index_tensor(self), shapes=ShapeContract.from_grid(model.grid, state),
            constants=EMPTY_MAPPING, boundary=boundary, kpp_forcing_inputs=kpp_forcing_inputs,
        )
        outcome = self.step_kernel(state, model, plan)
        # The counter advances HERE, in the eager epilogue that
        # runs immediately after ``_step_impl`` returns, instead of at the end
        # of the traced body.  Same moment in the step and the same value at
        # every read.  It runs BEFORE ``persist_step_outputs`` so the promoted
        # wrapper is cloned in the same pass as every other retained value.
        self.advance_step_index()
        return dataclasses.replace(
            outcome, state=self.persist_step_outputs(model, outcome.state),
        )

    def prepare_step_host(
        self,
        model: SplitStepperModelContext,
        state: PrognosticState,
        bundle: TimeStepBundle,
        *,
        time_seconds: float,
    ) -> "tuple[BoundaryPayload, KppStepForcing | None]":
        """Freeze host decisions and time samples before the traced step.

        The stepper's half of the prepare phase; ``build_step_plan``
        (``defgcm/step/plan.py``) calls it and publishes the result as a
        ``StepPlan``. This preparation runs outside the traced region.

        The returned tuple supplies the clocks' device payload
        (``StepPlan.boundary``) and KPP forcing inputs. Regional contexts resolve
        SamplingRule declarations through ``prepare_step_times``. The cube path returns an
        empty payload before regional sampling and never installs it.
        """
        key = (
            id(state), float(time_seconds), bundle,
            getattr(model, "_runner_sample_serial", 0),
        )
        from defgcm.accel import (
            StepGates,
            freeze_step_snapshot,
            step_snapshot_active,
        )
        # Local: ``defgcm.step`` imports the stepper package, so the module
        # level would close an import cycle.
        from defgcm.step.contract import BoundaryPayload

        if (
            getattr(self, "_prepared_step_key", None) == key
            and step_snapshot_active()
        ):
            # Idempotent: the same state and clock re-prepare to the same
            # payload, which is still installed.
            payload = self._prepared_boundary
            return payload, self._kpp_forcing_inputs

        composition = self._implicit_vertical_advection_composition(model)
        admission._validate_runtime_implicit_vertical_advection(
            composition,
            source="ImplicitFreeSurfaceStepper.prepare_step_host",
        )
        admission._refuse_flather_nonlinear_free_surface(self.nonlin_free_surf, model._find_flather_bc())
        admission._refuse_stepping_switches_unimplemented(
            mom_stepping=self.mom_stepping,
            open_boundary=(
                model.grid.boundary_topology.open_x  # type: ignore[attr-defined]
                or model.grid.boundary_topology.open_y  # type: ignore[attr-defined]
            ),
            nonhydrostatic=getattr(self.nonhydrostatic, "nonhydrostatic", False),
            external_pnh=self.external_pnh_provider is not None,
        )
        admission._refuse_r_star_grid(
            select_r_star=self.select_r_star,
            cube=model.grid.face_count == 6,  # type: ignore[attr-defined]
            open_boundary=(
                model.grid.boundary_topology.open_x  # type: ignore[attr-defined]
                or model.grid.boundary_topology.open_y  # type: ignore[attr-defined]
            ),
            has_hfac=ifs_geometry.surface_geometry_has_masks(model.grid.masks),  # type: ignore[attr-defined]
        )
        from defgcm.dynamics.vector_invariant_momentum import VectorInvariantMomentumAdvection

        admission._refuse_r_star_unread_ke_scheme(
            select_r_star=self.select_r_star,
            vector_invariant=self._resolved_vector_invariant_momentum,
            ke_schemes=tuple(module.select_ke_scheme for module in model.dynamics_modules
                             if isinstance(module, VectorInvariantMomentumAdvection)),
        )
        admit_cd_scheme(
            model.dynamics_modules, grid=model.grid,
            stagger_tracer_stepping=self.stagger_tracer_stepping,
            momentum_pressure_forcing=self.momentum_pressure_forcing,
            momentum_implicit_vertical_advection=self.momentum_implicit_vertical_advection,
        )
        self.stream_trace.step_index = self._step_index_value()
        # The two clocks, as host floats, for the step body's legacy branches
        # (a model context that takes no time sample, ``ClockSample.sampled``
        # false) and for the external phi_nh provider.  The traced step receives
        # the clock as a 0-d tensor (``StepPlan.times.current``); anything that needs a Python
        # float needs it decided here, in prepare, not converted back inside the
        # graph.
        object.__setattr__(self, "_prepared_current_seconds", float(time_seconds))
        object.__setattr__(
            self, "_prepared_future_seconds", float(time_seconds) + bundle.dt_clock
        )
        # The gW Adams-Bashforth factor uses the counter for one thing: is this
        # the first step.  Prepare makes that decision and publishes the value
        # ``_predict_vertical_momentum`` reads, which stops
        # changing after step 1; reading the raw counter in the traced step
        # installed a guard that failed every step.
        object.__setattr__(
            self,
            "_prepared_gw_my_iter",
            0 if self._step_index_value() == 0 else 1,
        )
        self._sample_external_pnh(model.grid)  # type: ignore[attr-defined]
        if composition.flags.active:
            # Kept out of ``_step_impl``: pure host
            # validation of the counter and of the cached momentum wrapper,
            # raising the same exceptions with the same messages.
            self._validate_step_index_state()
            admission._require_implicit_vertical_advection_execution_protocol(
                self._implicit_vertical_advection_execution(model),
            )
        gates = freeze_step_snapshot()
        object.__setattr__(self, "gates", StepGates.from_views(gates))
        self.prepare_step_runtime(model, model.grid, state)  # type: ignore[attr-defined]
        if (
            self._surface_dr_active(model.grid)  # type: ignore[attr-defined]
            and not getattr(self, "_nlfs_preconditions_checked", False)
        ):
            # Grid-only precondition with a host-side ``.any()``: once, eagerly.
            self._check_nonlin_free_surf_preconditions(model.grid)  # type: ignore[attr-defined]
            object.__setattr__(self, "_nlfs_preconditions_checked", True)
        # Dynamo cannot trace ``id() with unsupported args`` (once at
        # ``_vector_invariant_topology.py``).  Host-tuple constants owned
        # by a dynamics module -- today only the vector-invariant Coriolis
        # profile -- are resolved to tensors here, so the traced step reads a
        # memo keyed on ``(dtype, device)`` instead of building an ``id()`` key.
        for module in getattr(model, "dynamics_modules", ()):
            prepare_constants = getattr(module, "prepare_host_constants", None)
            if prepare_constants is not None:
                prepare_constants(
                    model.grid,  # type: ignore[attr-defined]
                    dtype=state.u.data.dtype,
                    device=state.u.data.device,
                )
        self._prewarm_helmholtz_coefficients(model, state, bundle)
        # The EOS dump tap resolves its environment on first use and builds a
        # lock the tracer cannot follow; resolving it here keeps that first use

        # of the first traced step).  Same process-once semantics, moved.
        from defgcm.dynamics.equation_of_state import resolve_eos_dump_tap

        resolve_eos_dump_tap()
        if getattr(self, "_resolved_3d_coriolis", None) is None:
            object.__setattr__(
                self,
                "_resolved_3d_coriolis",
                coriolis.resolve_3d_coriolis(
                    self.nonhydrostatic,
                    cell_area=model.grid.metrics.cell_area,
                    dx_center=model.grid.metrics.dx_center,
                    dy_center=model.grid.metrics.dy_center,
                ),  # type: ignore[attr-defined]
            )
        if composition.grid_kind == "cube":
            # Resolve the cube run's state-independent constants here, in the
            # eager phase, so the traced step only ever reads them back.
            #
            # It is the same call the step body makes, on the same arguments, so
            # whichever phase reaches it first gets the identical object -- what
            # this fixes is *which* phase that is.  The production runner warms
            # up zero eager steps by default (``runner.py`` ``eager_warmup_steps``),
            # so step 0 is the compiled step and the cold build landed inside the
            # traced region: ``_build_eager``'s fence broke the graph five times
            # on the synthetic cube and ten on cs32, and ``cube_tile_tables``'
            # ``setdefault`` once more.  Warm-memo steps never showed it, which
            # is why the gate (two eager steps) could not.
            from .implicit_free_surface_cube import resolve_cube_static_context

            resolve_cube_static_context(
                self,
                model.grid,  # type: ignore[attr-defined]
                getattr(model, "cube_ctx", None),
                state.temp.data.dtype,
                state.temp.data.device,
            )
            # Every cache key the traced step reads or finalize publishes exists
            # before the first traced call (histories and per-step readouts),
            # so the cache the tracer guards on step one is the steady-state
            # one; a restored stepper starts warm the same way.  Idempotent.
            from .implicit_free_surface_cube import seed_cube_cross_step_state

            seed_cube_cross_step_state(
                self,
                state,
                model.grid,  # type: ignore[attr-defined]
                bundle,
                getattr(model, "cube_ctx", None),
            )
            # The cube step body returns before the first install, so no clock
            # sample is taken and none is published.
            return self._publish_boundary(key, BoundaryPayload.empty())
        # Resolve the actual consumer before tracing; static-pressure EOS calls
        # do not consume the diagnostic history that HPG still publishes.
        object.__setattr__(self, "_uses_pressure_history", any(
            getattr(module, "dynamic_pressure", False)
            for module in model.dynamics_modules
        ))
        # The internal nonhydrostatic solve's mass-to-r-unit factor (1/rhoConst),
        # decided here as a host float so the traced step only reads it (a float()
        # of a model attribute in the trace can be a Tensor.item() graph break, gate
        # contract §7 rule 1).  Same condition as ``internal_nh_active`` in the step.
        # The vertical mixing schemes' N^2 reads the same factor
        # (pp81_ri_number.F:109-110, my82_ri_number.F:112-113), and so does the
        # conversion of the wind stress KPP and GGL90 read (fu*mass2rUnit,
        # external_forcing_surf.F:222), only when the assembly resolved a stress
        # producer (_package_modules.kpp_wind, read by _kpp_forcing_producers);
        # without one the surface momentum forcing is zero and reads no factor.
        reads_stress_factor = self.reads_kpp_forcing and self._package_modules.kpp_wind is not None
        if ((self.nonhydrostatic.nonhydrostatic and self.external_pnh_provider is None)
                or self.vertical_mixing_scheme != "none" or reads_stress_factor):
            object.__setattr__(self, "_prepared_mass_to_r_unit",
                               ifs_nh.pressure_mass_to_r_unit(model.dynamics_modules))
        if self._cached_total_phi_hyd is None:
            # The same seed, for a step that starts with no potential on the
            # stepper (see _seed_total_phi_hyd_picard): a resume from a
            # checkpoint that saved none, the direct model.step API, a step
            # after a window released its state.
            self._seed_total_phi_hyd_picard(model, state)
        columns.validate_vertical_mixing_composition(
            scheme=self.vertical_mixing_scheme, parameters=self.vertical_mixing_parameters,
            nz=model.grid.nz, modules=model.dynamics_modules,
        )
        if self.vertical_mixing_scheme == "ggl90":
            if self._ggl90_tke is None:
                object.__setattr__(self, "_ggl90_tke", columns.ggl90_cold_start_tke(
                    self.vertical_mixing_parameters, model.grid, like=state.temp.data))
        self._prepare_cg3d_coefficients(model.grid, state, bundle)  # type: ignore[attr-defined]
        prepare_times = getattr(model, "prepare_step_times", None)
        if prepare_times is None:
            raise TypeError(
                "regional step context requires prepare_step_times with SamplingRule declarations"
            )
        # ``reference`` fixes the dtype/device the traced step will ask the
        # OBCS balance for, so the eager sample can produce the balanced
        # targets too.
        # Each component declares its entry and application sampling clocks.
        from . import regional_cold_start

        regional_cold_start._hold_sponge_budget_as_tensors(model, state)
        regional_cold_start._prewarm_step_constants(self, model, state)
        payload = prepare_times(
            float(time_seconds),
            float(time_seconds) + bundle.dt_clock,
            reference=state.u.data,
        )
        # After prepare_times installed this step's forcing sample, so the check
        # reads the EmPmR the traced step will read; eager, every step.
        if (self.nonhydrostatic.nonhydrostatic and self.external_pnh_provider is None
                and self.use_real_freshwater_flux):
            ifs_nh.refuse_real_freshwater_pressure_term(
                model.dynamics_modules, model.grid,  # type: ignore[attr-defined]
                dtype=state.eta.data.dtype, device=state.eta.data.device,
            )
        regional_cold_start._seed_cross_step_state(
            self, model, state, dt_freesurf=bundle.dt_freesurf,
        )
        return self._publish_boundary(key, payload)

    def _prewarm_helmholtz_coefficients(
        self,
        model: SplitStepperModelContext,
        state: PrognosticState,
        bundle: TimeStepBundle,
    ) -> None:
        """Fill the Helmholtz coefficient memo here, so the kernel only reads it.

        ``_get_coefficients`` is called from inside the traced step, and on a
        miss it built the triple there, inside ``with constant_transport()`` --
        a context manager Dynamo declines to inline (``SKIPPED INLINING
        __enter__``, one graph break).  Running the same build on the same key in prepare leaves the
        traced call a dictionary hit. Linear runs use the same geometry and
        expression. For nlfs=1/2, the first build must precede the initial
        surface-thickness writes: MIT INITIALISE_FIXED:223 calls INI_CG2D
        with resting hFac and never updates that matrix at these levels.

        The key's varying fields are all decided here.
        ``dt_mom``/``dt_freesurf`` (from ``bundle``, which is what the kernel
        receives as ``plan.times.bundle``) and ``alpha_beta`` are the same
        objects and values the step body will present; the dtype/device come
        off ``state.eta.data``, which is what the regional step body passes
        (``bounded.eta.data`` -- same dtype and device as the prognostic η it
        is derived from). On linear geometry a different key rebuilds from the
        same geometry. Nonlinear levels 1/2 require the original coefficient
        keys throughout the fixed-time-step continuation.

        Two postures decline:

        * ``nonlin_free_surf > 2`` rebuilds the coefficients every step from
          geometry the step body itself rewrites, and ``_get_coefficients``
          bypasses its cache there (MITgcm ``UPDATE_CG2D``,
          ``forward_step.F:869`` at MITgcm 27256b2d).  Prewarming would build a triple nobody
          reads.
        * Once nlfs=1/2 geometry is initialized, this hook must not build from
          the lagged grid layer. The solve reuses its first coefficients;
          whole-model restore installs them together with the two hFac layers.
        """
        if self.nonlin_free_surf > 2 or (
            self.nonlin_free_surf >= 1 and self._surface_geometry.surface is not None
        ):
            return
        grid = model.grid  # type: ignore[attr-defined]
        # The same two reads the two step bodies make: the regional one passes
        # ``ctx=None`` (``implicit_free_surface.py`` ``_solve_free_surface``
        # call) and the cube one reads ``model.cube_ctx``
        # (``implicit_free_surface_cube.py:1794``), which is ``None`` off the
        # multi-rank harness.  The dtype/device half of the key is read off the
        # field each body keys on: the cube entry takes ``u.dtype``/``u.device``
        # (``_prepare_cube_step_entry``), the regional solve the bounded η,
        # which shares u's dtype and device (a key read off a
        # different field would miss in the kernel and add a second entry to
        # the grid runtime cache, moving its ``len`` guard).
        is_cube = grid.face_count == 6
        ctx = getattr(model, "cube_ctx", None) if is_cube else None
        keyed_on = state.u.data if is_cube else state.eta.data
        self._get_coefficients(
            grid,
            bundle.dt_mom,
            bundle.dt_freesurf,
            keyed_on.dtype,
            keyed_on.device,
            ctx=ctx,
        )

    def _initial_eta_h(
        self, state, grid, dt_freesurf, *, w_interface_to_down=None,
    ) -> torch.Tensor:
        eta = state.eta.data
        if not self.exact_conserv or self.implicit_div_2d_flow == 1.0:
            return eta
        if w_interface_to_down is None:
            from defgcm.dynamics.vertical_velocity import diagnose_vertical_velocity

            _, w_interface_to_down, _ = diagnose_vertical_velocity(state, grid)
        return ifs_freshwater.initial_continuity_surface(
            eta=eta, beta=self.implicit_div_2d_flow,
            w_interface_to_down=w_interface_to_down, dt_freesurf=dt_freesurf,
        )

    def _evaluation_inputs(self, model, state, grid, hydro_history, *,
                           keep: bool = True,
                           pmep_r: torch.Tensor | None = None,
                           kappa_ru_floor: torch.Tensor | None = None,
                           kappa_rv_floor: torch.Tensor | None = None,
                           coriolis_u: torch.Tensor | None = None,
                           coriolis_v: torch.Tensor | None = None) -> EvaluationInputs:
        """The slow-tendency inputs of this step, passed to every evaluation.

        Nothing is written to the grid: the lagged w and the lagged total
        hydrostatic potential are state of the step, and the grid outlives
        every step.  The stepper keeps this step's two tensors for
        :meth:`evaluated_inputs` (diagnostics read the inputs of the last
        evaluated step) until the next step or the end of the checkpoint
        segment (``release_working_step_state``).
        """
        w_interface_prev = (hydro_history.w_interface if hydro_history is not None
                            else self._continuity.w_interface_prev)
        if w_interface_prev is None:
            # A fresh run has no previous step.  MITgcm's INITIALISE_VARIA calls
            # INTEGR_CONTINUITY on the initial u, v, after OBCS_INIT_VARIABLES has
            # applied the open boundaries, for every run, hydrostatic included
            # (initialise_varia.F:263,334-335; integr_continuity.F:267-274), so the
            # first DYNAMICS reads that wVel.  Diagnose it the same way from
            # ``state``, which the step passes bounded; the momentum modules raise
            # on a missing w (the explicit-input requirement).
            w_interface_prev = _carry(prepare.diagnose_mit_interface_velocity(state=state, grid=grid))
        # Grid-constant resolution (a ``torch.equal`` on metrics = device sync +
        # graph break in the body): computed once in ``prepare_step_host``.
        resolved_3d_coriolis = getattr(self, "_resolved_3d_coriolis", None)
        if resolved_3d_coriolis is None:
            resolved_3d_coriolis = coriolis.resolve_3d_coriolis(
                self.nonhydrostatic,
                cell_area=grid.metrics.cell_area,
                dx_center=grid.metrics.dx_center,
                dy_center=grid.metrics.dy_center,
            )
            object.__setattr__(self, "_resolved_3d_coriolis", resolved_3d_coriolis)
        # Lagged total hydrostatic potential for the JMD95P EOS.  The EOS
        # expects MIT-style previous-step totPhiHyd; the current step's HPG
        # diagnostic refreshes this cache below for the next step.
        # Only the dynamic-pressure EOS consumes this history. Static-pressure
        # EOS calls must not carry a first-step None/later Tensor through the
        # compiled evaluation merely because HPG publishes a diagnostic.
        total_phi_hyd = (self._cached_total_phi_hyd
                         if hydro_history is None and self._uses_pressure_history else None)
        seaice = self._package_modules.seaice
        if keep:
            object.__setattr__(self, "_evaluated_w_interface", w_interface_prev)
            object.__setattr__(self, "_evaluated_total_phi_hyd", total_phi_hyd)
            if pmep_r is not None:
                object.__setattr__(self, "_evaluated_pmep_r", pmep_r)
        return EvaluationInputs(
            w_interface=w_interface_prev,
            total_phi_hyd=total_phi_hyd,
            select_3d_cori_scheme=resolved_3d_coriolis.select_3d_cori_scheme,
            f_cori_cos=resolved_3d_coriolis.f_cori_cos,
            pmep_r=pmep_r,
            kappa_ru_floor=kappa_ru_floor,
            kappa_rv_floor=kappa_rv_floor,
            coriolis_u=coriolis_u,
            coriolis_v=coriolis_v,
            ice_area=None if seaice is None else seaice.published.ice_area,
            momentum_pressure_forcing=self.momentum_pressure_forcing,
        )

    def evaluated_inputs(self) -> EvaluationInputs | None:
        """The inputs of the last evaluated step, for diagnostics. **Eager.**"""
        resolved = getattr(self, "_resolved_3d_coriolis", None)
        w_interface = getattr(self, "_evaluated_w_interface", None)
        total_phi_hyd = getattr(self, "_evaluated_total_phi_hyd", None)
        pmep_r = getattr(self, "_evaluated_pmep_r", None)
        if resolved is None and w_interface is None and total_phi_hyd is None:
            return None
        return EvaluationInputs(
            w_interface=w_interface,
            total_phi_hyd=total_phi_hyd,
            select_3d_cori_scheme=None if resolved is None else resolved.select_3d_cori_scheme,
            f_cori_cos=None if resolved is None else resolved.f_cori_cos,
            pmep_r=pmep_r,
            momentum_pressure_forcing=self.momentum_pressure_forcing,
        )

    def _refresh_hfac_surf(self, grid, eta_h_new, *, obc_eta):
        resting = self._surface_geometry.resting
        surface = ifs_geometry.refresh_hfac_surf(
            grid, eta_h_new, resting=resting, hfac_inf=self.hfac_inf, obc_eta=obc_eta,
        )
        return ifs_geometry.SurfaceGeometry(resting, surface)

    def _apply_surf_dr(self, grid):
        surface_hfac = self._surface_geometry.surface
        if surface_hfac is not None:
            ifs_geometry.apply_surf_dr(
                grid,
                resting=self._surface_geometry.resting,
                surface=surface_hfac,
            )



    def _predict_vertical_momentum(
        self, bounded, *, w_vel_mit, phi_nh_prev, gw_prev, dt_mom, grid, model,
        diagnostics, kpp_fields, resolved_3d_coriolis,
    ):
        (
            viscAh_W_for_gw, kappa_ru_for_gw, kappa_rv_for_gw, viscA4_W_for_gw,
            no_slip_sides_for_gw, side_drag_factor_for_gw,
        ) = ifs_nh.vertical_momentum_viscosity_inputs(
            modules=getattr(model, "dynamics_modules", ()), diagnostics=diagnostics,
            bounded=bounded, w_vel_mit=w_vel_mit, grid=grid, kpp_fields=kpp_fields,
            implicit_viscosity=self.implicit_viscosity,
        )

        # compute_gw consumes MIT upward-positive w_vel_mit after
        # first-step/cache conversion. MIT CALC_GW reads start-of-step
        # uVel/vVel, not the post-tendency u_star/v_star predictors.
        gw_current, gw_diss = compute_gw(
            bounded,
            w_vel_mit=w_vel_mit,
            grid=grid,
            params=self.nonhydrostatic,
            mom_advection=self.nh_mom_advection,
            mom_viscosity=self.nh_mom_viscosity,
            no_slip_sides=no_slip_sides_for_gw,
            side_drag_factor=side_drag_factor_for_gw,
            kappa_ru=kappa_ru_for_gw,
            kappa_rv=kappa_rv_for_gw,
            viscAh3d_00=viscAh_W_for_gw,
            viscA4_W=viscA4_W_for_gw,
            select_3d_cori_scheme=(
                resolved_3d_coriolis.select_3d_cori_scheme
            ),
            f_cori_cos=resolved_3d_coriolis.f_cori_cos,
        )
        gw_prev = torch.zeros_like(w_vel_mit) if gw_prev is None else gw_prev.to(
            dtype=w_vel_mit.dtype,
            device=w_vel_mit.device,
        )
        if gw_prev.shape != w_vel_mit.shape:
            raise ValueError("gwNm1_prev must match w_vel_mit")
        # MITgcm ADAMS_BASHFORTH2 (adams_bashforth2.F:61-65): abFac = 0 on the first step of
        # a cold start, else 0.5 + abEps.  LOCOMgcm's ab_eps <= 0 steps u, v, T and S forward in
        # time without extrapolation (the stepper's ``ab_eps > 0`` gates), which is
        # MITgcm's abEps = -0.5: abFac = 0 for gW as well.  compute_gw masks both
        # terms and the history holds their masked sum, so the extrapolation keeps
        # dry W cells at zero without another mask.
        if self._prepared_gw_my_iter == 0 or float(self.ab_eps) <= 0.0:
            ab_fac = 0.0
        else:
            ab_fac = ifs_tendency.ab2_factor(self.ab_eps)
        tendency_for_history = gw_current + gw_diss if self.mom_dissip_in_ab else gw_current
        gw_ab = ifs_tendency.extrapolate_ab2(tendency_for_history, gw_prev, factor=ab_fac)
        if not self.mom_dissip_in_ab:
            gw_ab = gw_ab + gw_diss
        gw_nm1_history = _carry(tendency_for_history)
        # TIMESTEP_WVEL produces MIT upward-positive transient wVel for
        # the same-step PRE_CG3D RHS assembly.
        w_vel_new_mit_for_cg3d, _ = timestep_wvel(
            w_vel_mit,
            gw_ab,
            phi_nh_prev,
            grid,
            dt_mom,
            self.nonhydrostatic,
            # implicitNHPress defaults to implicSurfPress (ini_parms.F:485-486).
            implicit_nh_press=self.implicit_surf_press,
        )
        return w_vel_new_mit_for_cg3d, gw_nm1_history

    def _predict_velocity(
        self, u, v, eta, tend_u, tend_v, grid, *, dt_mom, alpha, field_viscosity, dtype, device,
        hFacW=None, hFacS=None, u_mask=None, v_mask=None, ctx=None, momentum_w=None,
    ):
        """Velocity predictor u*: AB2/PG RHS then one vertical implicit solve.

        When momentum implicit vertical advection is enabled, ``momentum_w`` must
        be the previous-half-step ``MomentumWNmHalfMIT`` wrapper: complete physical
        velocity in MITgcm upward-positive convention.  U and V each assemble
        viscosity plus advection into one matrix and perform one strict solve.

        MITgcm: ``gUdPx = -phFac*dPhiHydX - psFac*phiSurfX`` where
        ``psFac = pfFacMom * (1 - implicSurfPress)`` — the explicit fraction (1-α) of
        the surface PG is added in the predict; the implicit vertical viscosity is
        the IMPLDIFF pass that follows TIMESTEP (``dynamics.F:587-591`` for U with
        ``recip_hFacW``; ``dynamics.F:595-599`` for V with ``recip_hFacS``).

        ``field_viscosity`` is an **explicit** argument carrying the face
        fields ``kappa_ru``/``kappa_rv``, or None for the scalar background: the
        regional caller passes its assembled step coefficients and the cube
        caller passes ``_cube_kpp``.

        Faithfulness notes:

        * PG coefficient chain: the regional source added
          ``+(-explicit_frac · g_val_tensor · dt_mom · grad.unsqueeze(1).expand_as(u*))``
          while cube subtracted ``explicit_frac · gravity · dt_mom · grad.unsqueeze(1)``
          (float scalar chain, broadcast).  These are bitwise-equal: IEEE
          round-to-nearest is sign-symmetric (``fl(-a·b) == -fl(a·b)`` and
          ``x + (-y) == x - y``), the scalar grouping ``((ef·g)·dt)·grad`` is the same
          in both, ``expand_as`` vs broadcast produce the identical elementwise
          product, and in the aligned float64 path every scalar intermediate is
          float64 regardless of tensor-vs-float packaging.  Unified to the cube form.

        * ``recip_hFac``: unified to the regional *double-where* construction
          (``mask = hf>0; safe = where(mask, hf, 1); recip = where(mask, 1/safe, 0)``).
          Forward-identical to cube's single-where direct divide
          (``where(hf>0, 1/hf, 0)``): where ``hf>0`` both yield ``1/hf`` (``safe==hf``
          there), where ``hf==0`` both yield ``0``.  The double-where additionally
          keeps the backward pass NaN-free at dry cells (never divides by 0) — a
          declared improvement; the gate judges forward only.

        * Mask position: masks fire only when ``u_mask is not None`` — after
          the explicit PG, and after IMPLDIFF (the latter only when IMPLDIFF ran,
          matching the cube source exactly: ``PG -> mask -> IMPLDIFF -> mask``).
          Regional passes ``u_mask=None`` so neither fires (its masking lives in the
          out-of-segment ``has_3d_masks`` block).
        """
        gravity = self.gravity
        momentum_implicit = getattr(
            self, "momentum_implicit_vertical_advection", False
        )
        if type(momentum_implicit) is not bool:
            raise TypeError(
                "stepper.momentum_implicit_vertical_advection must be an exact bool"
            )

        # 2. Predict: u* = u^n + dt * (AB2_tend - (1-α)·g·∇η^n)
        #    The enabled implicit-advection branch follows TIMESTEP exactly: old
        #    velocity is outside maskW/S, while every tendency term is inside it.
        #    The all-off branch retains the historical arithmetic verbatim.
        if momentum_implicit:
            hfw_for_mask = hFacW if hFacW is not None else grid.masks.hFacW
            hfs_for_mask = hFacS if hFacS is not None else grid.masks.hFacS
            predictor_mask_u, predictor_mask_v = make_momentum_predictor_masks(
                u, v, hfw_for_mask=hfw_for_mask, hfs_for_mask=hfs_for_mask,
            )
            u_star, v_star = compute_velocity_predictor(
                u, v, tend_u, tend_v, dt_mom=dt_mom,
                predictor_mask_u=predictor_mask_u, predictor_mask_v=predictor_mask_v,
            )
        else:
            u_star, v_star = compute_unmasked_velocity_predictor(
                u=u, v=v, tend_u=tend_u, tend_v=tend_v, dt_mom=dt_mom,
            )
        self.stream_trace.emit_stream_trace(
            "cube.predictor_after_ab",
            u_star=u_star,
            v_star=v_star,
        )

        explicit_frac = 1.0 - alpha if self.momentum_pressure_forcing else 0.0
        if explicit_frac > 0.0:
            # Add explicit surface pressure gradient: -(1-α)·g·dt·∇η^n
            _dist_trace("explicit surface pressure gradient begin")
            geu, gev = surface_pressure_gradients(eta, grid, dtype=dtype, device=device, ctx=ctx)
            if momentum_implicit:
                u_star, v_star = apply_explicit_surface_pressure(
                    u_star, v_star, explicit_frac=explicit_frac, gravity=gravity, dt_mom=dt_mom,
                    geu=geu, gev=gev, predictor_mask_u=predictor_mask_u, predictor_mask_v=predictor_mask_v,
                )
            else:
                u_star, v_star = apply_unmasked_explicit_surface_pressure(
                    u_star=u_star, v_star=v_star, explicit_frac=explicit_frac,
                    gravity=gravity, dt_mom=dt_mom, geu=geu, gev=gev,
                )
            _dist_trace("explicit surface pressure gradient done")
            self.stream_trace.emit_stream_trace(
                "cube.explicit_surface_pressure_gradient",
                eta=eta,
                phiSurfX_explicit=gravity * geu,
                phiSurfY_explicit=gravity * gev,
                u_star=u_star,
                v_star=v_star,
            )
        if u_mask is not None and not momentum_implicit:
            u_star, v_star = mask_velocity_pair(u_star, v_star, u_mask=u_mask, v_mask=v_mask)
        self.stream_trace.emit_stream_trace(
            "cube.predictor_after_explicit_pg_mask",
            u_star=u_star,
            v_star=v_star,
        )

        # 2b. Implicit vertical viscosity (MITgcm: IMPLDIFF after TIMESTEP)
        #     Solves (I - dt·K·∂²/∂z²) u*_new = u*_predict for each velocity
        #     component, replacing the explicit VerticalMixing momentum tendency.
        #     MITgcm: recip_hFacW for U (dynamics.F:587-591),
        #             recip_hFacS for V (dynamics.F:595-599).
        # KPP-on: field viscosity KappaRU/RV (cell-centre KPPviscAz -> u/v faces);
        # else scalar stepper.implicit_viscosity.  The implicit solver accepts both.
        viscAr = float(self.implicit_viscosity)
        implicit_viscosity = bool(
            getattr(self, "implicit_vertical_viscosity", True)
        )
        if momentum_implicit:
            upwind_shear = getattr(self, "upwind_shear", None)
            if type(upwind_shear) is not bool:
                raise TypeError(
                    "stepper.upwind_shear must be an exact bool when momentum "
                    "implicit vertical advection is enabled"
                )
            if momentum_w is None:
                raise ValueError(
                    "momentum implicit vertical advection requires MomentumWNmHalfMIT"
                )
            from defgcm.dynamics.implicit_vertical_advection import (
                solve_u_implicit_vertical_advection,
                solve_v_implicit_vertical_advection,
            )

            solve_grid = grid
            if grid.face_count == 6 and ctx is not None:
                from defgcm.geometry.cube_localize import localize_cube_grid

                solve_grid = localize_cube_grid(grid, ctx)
            if implicit_viscosity:
                visc_u = field_viscosity.kappa_ru if field_viscosity is not None else viscAr
                visc_v = field_viscosity.kappa_rv if field_viscosity is not None else viscAr
            else:
                visc_u = 0.0
                visc_v = 0.0
            u_star = solve_u_implicit_vertical_advection(
                rhs_u=u_star,
                w=momentum_w,
                dt_mom=dt_mom,
                viscosity=visc_u,
                grid=solve_grid,
                upwind_shear=upwind_shear,
            )
            v_star = solve_v_implicit_vertical_advection(
                rhs_v=v_star,
                w=momentum_w,
                dt_mom=dt_mom,
                viscosity=visc_v,
                grid=solve_grid,
                upwind_shear=upwind_shear,
            )
        elif implicit_viscosity and (field_viscosity is not None or viscAr > 0.0):
            # recip_hFac: double-where.  hFacW/S passed None -> fall back to
            # grid.masks (regional; may still be None -> recip stays None); passed
            # tensor -> ``.to`` is idempotent (cube already at dtype/device).
            hfw_in = hFacW if hFacW is not None else grid.masks.hFacW
            hfs_in = hFacS if hFacS is not None else grid.masks.hFacS
            recip_hFacW, recip_hFacS = reciprocal_velocity_hfacs(
                u_star, v_star, hfw_in=hfw_in, hfs_in=hfs_in,
            )
            visc_u = field_viscosity.kappa_ru if field_viscosity is not None else viscAr
            visc_v = field_viscosity.kappa_rv if field_viscosity is not None else viscAr
            u_star = columns.solve_implicit_vertical_diffusion(
                field=u_star, layer_thickness=grid.layer_thickness,
                interface_spacing=grid.vertical_coordinate.interface_spacing, dt=dt_mom,
                diffusivity=visc_u, recip_hfac_matrix=recip_hFacW,
            )
            v_star = columns.solve_implicit_vertical_diffusion(
                field=v_star, layer_thickness=grid.layer_thickness,
                interface_spacing=grid.vertical_coordinate.interface_spacing, dt=dt_mom,
                diffusivity=visc_v, recip_hfac_matrix=recip_hFacS,
            )
            # dynamics.F:614-633: the CD scheme's D-grid velocity takes the same
            # IMPLDIFF, vVelD (U points) with kappaRU and recip_hFacW, uVelD
            # (V points) with kappaRV and recip_hFacS.  The scheme replaced
            # _cd_state earlier in this step, so it is present exactly when the
            # composition has one.
            cd_state = self._cd_state
            if cd_state is not None:
                object.__setattr__(self, "_cd_state", dataclasses.replace(
                    cd_state,
                    v_d=columns.solve_implicit_vertical_diffusion(
                        field=cd_state.v_d, layer_thickness=grid.layer_thickness,
                        interface_spacing=grid.vertical_coordinate.interface_spacing, dt=dt_mom,
                        diffusivity=visc_u, recip_hfac_matrix=recip_hFacW,
                    ),
                    u_d=columns.solve_implicit_vertical_diffusion(
                        field=cd_state.u_d, layer_thickness=grid.layer_thickness,
                        interface_spacing=grid.vertical_coordinate.interface_spacing, dt=dt_mom,
                        diffusivity=visc_v, recip_hfac_matrix=recip_hFacS,
                    ),
                ))
            if u_mask is not None:
                u_star, v_star = mask_velocity_pair(u_star, v_star, u_mask=u_mask, v_mask=v_mask)
        self.stream_trace.emit_stream_trace(
            "cube.predictor_after_impldiff",
            u_star=u_star,
            v_star=v_star,
        )

        return u_star, v_star


    def _solve_free_surface(
        self, eta, etaH_prev, div_star, grid, *, dt_mom, dt_freesurf, beta,
        dtype, device, eta_source, rA=None, surface_mask=None, mask_in_c=None,
        x0_mask=None, use_real_freshwater_flux, empmr_mass_flux,
        free_surf_fac=1.0, mass_to_r_unit=1.0, ctx=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helmholtz RHS assembly + cg2d solve for η^{n+1}.

        Conceptually, without real freshwater,
        ``[-α·β·g·dt_mom·dt_freesurf·∇·(H·∇) + I] η^{n+1} = etaH - β·dt_freesurf·div(U*)``.
        With ``use_real_freshwater_flux``, the integrated RHS additionally subtracts
        ``freeSurfFac·mass2rUnit·β·dt_freesurf·rA·EmPmR``.
       The matrix is assembled in MITgcm's *integrated* form (``rA`` on the
        diagonal, no ``/rA`` on the off-diagonals), so the RHS carries an ``rA``
        factor too and both sides scale consistently.  MITgcm
        ``solve_for_pressure.F`` uses ``etaH`` (not ``etaN``) for the mass term when
        ``exactConserv=.TRUE.``; with β=1 the two collapse.

        ``eta_source`` is the mass source chosen by the time program: etaH,
        or etaH plus the internal NH surface pressure under oldFreeSurfTerm.

        ``use_real_freshwater_flux`` and ``empmr_mass_flux`` are required.
        ``empmr_mass_flux=None`` declares that no EmPmR field reaches
        this solve; together with ``use_real_freshwater_flux=True`` it raises,
        because MITgcm adds the configured EmPmR to ``cg2d_b``
        (``solve_for_pressure.F:146-153``) and a zero field would silently remove
        that term.

        ``etaH_prev`` is an **explicit** argument: each caller reads its own history
        (regional explicit eta_h history or η, cube via its step entry), and
        regional and cube pressure readouts leave through StepOutcome.

        Faithfulness notes:

        * **x0 policy is three-state and load-bearing**: the initial guess is
          ``eta·x0_mask`` when a mask is given, else bare ``eta``.  Regional-with-OB
          passes ``maskInC``, regional-without-OB passes ``None`` (its dry cells keep
          η, matching MITgcm ``update_etah.F``), the cube passes ``surf`` (dry cells
          forced to 0). The three do **not** converge: the excluded rows are exactly
          decoupled (``aC==rA``, zero incidence, rhs=0), yet the different guess
          changes the PCG global scalars, so the wet-cell solution moves ~3e-16 and
          the iteration count changes (measured 42 vs 44).  Merging any two of them
          would change numbers.

        * Surface mask: both callers pass an **already-sliced** k=0 centre wet
          mask (regional ``center_float(...)[0:1]`` keeping the batch axis, cube
          ``surf``); the two are bitwise the same ``hFacC>0`` layer. The segment
          never re-indexes it, so neither caller's index form leaks in here.
          ``None`` = do not mask (regional without 3-D masks).

        * OB interior mask: regional-only; applied to the RHS *after* the
          surface mask, as a second multiply (MITgcm ``cg2d_b *= maskInC``).  Cube
          passes ``None``.

        * ``rA``: ``None`` means "read it off the grid here", which reproduces the
          regional form ``cell_area_for(...).unsqueeze(0)`` at the η history's
          dtype/device; the cube always passes its (possibly rank-local) ``rA``.

        * ``ctx``: forwarded to ``_get_coefficients`` (rank enters the coefficient
          cache key) and to ``cg2d_solve`` (distributed matvec + face-ordered dot).
         The regional caller passes ``None``, which is the byte-identical
          single-rank path in both leaves.

        * Cube batch axis: ``cg2d_solve`` wants a leading batch axis, which
          the regional tensors already have and the cube ones do not — hence the
          ``unsqueeze(0)`` / ``[0]`` pair on the cube branch only.

        * Traces: both callers' traces are kept and now fire on both paths —
          the two regional ``_dist_trace`` strings bracketing the solve and the cube
          ``cube.cg2d_rhs`` / ``cube.cg2d_solution`` stream traces.  Cross-emission
          is numerically inert (``StreamTrace.emit_stream_trace`` is a no-op without an
          installed hook, ``_dist_trace`` is an env-gated print).
        """
        is_cube = grid.face_count == 6
        if rA is None:
            # [1, ny, nx] — broadcasts over batch.  dtype/device come off the η
            # history rather than the ``dtype``/``device`` arguments so this stays the
            # regional source's exact allocation (they agree whenever the step runs in
            # one dtype, which the aligned path does).
            rA = grid.metrics.cell_area_for(
                etaH_prev.dtype, etaH_prev.device,
            ).unsqueeze(0)
        rhs = compute_free_surface_rhs(
            rA=rA, eta_source=eta_source, beta=beta,
            dt_freesurf=dt_freesurf, div_star=div_star,
        )
        if use_real_freshwater_flux:
            if empmr_mass_flux is None:
                raise ValueError(
                    "use_real_freshwater_flux=True but empmr_mass_flux is None: the "
                    "real-freshwater cg2d source term (solve_for_pressure.F:146-153) "
                    "has no EmPmR input; wire the freshwater forcing or turn the "
                    "switch off"
                )
            rhs = subtract_real_freshwater_rhs(
                rhs=rhs, empmr_mass_flux=empmr_mass_flux, rA=rA,
                free_surf_fac=free_surf_fac,
                mass_to_r_unit=mass_to_r_unit,
                implic_div_2d_flow=beta,
                dt_freesurf=dt_freesurf,
            )

        rhs, x0 = prepare_free_surface_solve(
            rhs=rhs, eta=eta, surface_mask=surface_mask, mask_in_c=mask_in_c,
            x0_mask=x0_mask, is_cube=is_cube, stream_trace=self.stream_trace,
        )
        coeffs = self._get_coefficients(
            grid, dt_mom, dt_freesurf, dtype, device, ctx=ctx,
        )
        tolerance = self.cg_tolerance
        tolerance_mode = getattr(self, "cg_tolerance_mode", None)
        if tolerance_mode == "mit_w_unit":
            layout = getattr(grid, "parallel_layout", None)
            tolerance = cg2d_w_unit_tolerance(
                grid, target_res_w_unit=self.cg_tolerance,
                implicit_div_2d_flow=beta, dt_freesurf=dt_freesurf, rA=rA,
                surface_mask=surface_mask, mask_in_c=mask_in_c,
                distributed=ctx is not None or (layout is not None and layout.world_size > 1),
            )
            tolerance_mode = "absolute"
        with graph_exit():  # The PCG host loop, whole
            eta_new, num_iters_t, residual_t = _solve_cg2d_and_stage(
                self.stream_trace,
                rhs,
                x0,  # initial guess (masked per the caller's x0 policy)
                coeffs,
                grid,
                is_cube=is_cube,
                max_iters=self.cg_max_iters,
                tolerance=tolerance,
                tolerance_mode=tolerance_mode,
                use_min_res_sol=self.cg_use_min_res_sol,
                ctx=ctx,
            )
        return eta_new, num_iters_t, residual_t

    def _correct_surface_velocity(
        self, eta_new, u_star, v_star, grid, *, dt_mom, alpha, dtype, device,
        wet_face_mask_u=None, wet_face_mask_v=None, mask_in_w=None, mask_in_s=None,
        u_mask=None, v_mask=None, ctx=None,
    ):
        """Velocity correction: u^{n+1} = u* - α·g·dt·∇η^{n+1} (implicit surface PG).

        MITgcm ``correction_step.F``: ``psFac = pfFacMom * implicSurfPress`` and
        ``uVel = gU - psFac*dt*phiSurfX`` with ``phiSurfX = g·∂η^{n+1}/∂x`` multiplied
        by ``_maskW`` / ``maskInW``.  Only the implicit fraction α of the surface PG is
        applied here — the explicit fraction (1-α) was already added by
        ``_predict_velocity``.

        Like the predictor segment, this one has **no w / vertical-velocity or depth
        input**: the only sign in play is the surface-PG coefficient, which is
        topology- and convention-neutral.

        The three optional mask groups are independent, and keeping them independent
        is what preserves both callers' (differing) masking semantics:

        * ``wet_face_mask_u/v`` + ``mask_in_w/s`` — regional 2-D face masks applied to
          the η *gradient*.  Zeroing the gradient leaves a dry / open-boundary face at
          its predictor value ``u*``.
        * ``u_mask`` / ``v_mask`` — cube masks applied to the *result*, so a dry face
          ends at 0, not at ``u*``.

        These two semantics must **not** be merged: converging them would change
        numbers on one side.  Each caller passes one group, never both.

        Faithfulness notes:

        * Coefficient chain: the regional source built a 0-d tensor
          ``torch.tensor(α·g·dt_mom, dtype=u*.dtype, device=u*.device)`` and scaled
          ``grad.unsqueeze(1).expand_as(u*)``; the cube source used the float scalar
          chain ``α·gravity·dt_mom * grad.unsqueeze(1)`` with plain broadcast.
          Unified to the cube form: the left-associated ``((α·g)·dt_mom)`` product is
          the same python float in both, a python float multiplied against a floating
          tensor is wrapped at that tensor's dtype (same value as the explicit
          ``torch.tensor(..., dtype=...)``), and ``expand_as`` vs broadcast produce the
          identical elementwise product (the same argument as for the predictor).
          This relies on ``u_star.dtype == grad.dtype`` — true because the gradient
          leaves allocate their output at the η field's dtype and the whole step runs
          in one dtype; a mixed-dtype caller would round the scalar at the *gradient*
          dtype rather than the velocity's, so keep that invariant if either changes.

        * Regional mask order: wet-face mask first, OB mask second, as two
          separate multiplies — not fused into one combined mask.

        * Leaf dispatch: ``face_count == 6`` selects
          ``_cube_eta_gradient`` (``ctx`` routes the distributed scalar exchange and
          the dxC/dyC metric scatter); otherwise the regional ``gradient_eta_to_u/v``
          pair, which take only ``(eta, grid)``.  ``dtype``/``device`` feed the cube
          leaf only.

        * Traces: both callers' traces are kept, so each now fires on both
          paths — the two regional ``_dist_trace`` strings bracketing the leaf, and the
          cube ``cube.correction`` stream trace at the end with its original kwargs.
          Cross-emission is numerically inert (``_emit_stream_trace`` is a no-op
          without an installed hook; its ``gravity * grad`` kwargs are throw-away
          reads of the same gradients).
        """
        gravity = self.gravity
        _dist_trace("eta correction gradients begin")
        grad_u, grad_v = surface_pressure_gradients(eta_new, grid, dtype=dtype, device=device, ctx=ctx)
        _dist_trace("eta correction gradients done")

        # Gradient masks (regional): wet-face first, then the OB interior mask.
        grad_u, grad_v = mask_surface_pressure_gradients(
            grad_u, grad_v, wet_face_mask_u=wet_face_mask_u, wet_face_mask_v=wet_face_mask_v,
            mask_in_w=mask_in_w, mask_in_s=mask_in_s,
        )

        # correction_step.F:152: pfFacMom applies on both regional and cube grids.
        if not self.momentum_pressure_forcing:
            alpha = 0.0
        u_new, v_new = compute_velocity_correction(
            u_star, v_star, alpha=alpha, gravity=gravity, dt_mom=dt_mom, grad_u=grad_u, grad_v=grad_v,
        )
        # Result mask (cube): dry face -> 0.  Regional passes None.
        if u_mask is not None:
            u_new, v_new = mask_velocity_pair(u_new, v_new, u_mask=u_mask, v_mask=v_mask)
        self.stream_trace.emit_stream_trace(
            "cube.correction",
            phiSurfX=gravity * grad_u,
            phiSurfY=gravity * grad_v,
            u_new=u_new,
            v_new=v_new,
        )
        return u_new, v_new

    def _tendency_evaluation_sources(self, model, modules) -> ifs_tendency.TendencySources:
        return ifs_tendency.TendencySources(
            model.grid, modules,
            model._populate_module_diagnostics, model.boundary_condition.enforce_tendency,
        )

    def _tracer_tendency_sources(self, model) -> ifs_tendency.TendencySources:
        modules = tuple(model.slow_tracer_modules)
        return self._tendency_evaluation_sources(model, modules)

    def _evaluate_slow_tendency_phase(self, model, state, *, kpp, assembly_policy, inputs):
        """``(ab2_tendency, post_ab2_tendency, diagnostics)`` of the step's slow
        tendencies (momentum phase only when tracers are staggered); one
        dispatch for the step and the prepare-time prewarm."""
        if self.stagger_tracer_stepping and self.momentum_implicit_vertical_advection:
            return ifs_tendency._evaluate_momentum_tendency_for_implicit_vertical_advection(
                       state,
                       sources=self._tendency_evaluation_sources(model, tuple(model.slow_momentum_modules)),
                       kpp=kpp,
                       assembly_policy=assembly_policy,
                       inputs=inputs,
                   )
        if self.stagger_tracer_stepping and hasattr(model, 'evaluate_momentum_tendency_split_ab2'):
            return model.evaluate_momentum_tendency_split_ab2(
                state, assembly_policy=assembly_policy, inputs=inputs)
        if self.stagger_tracer_stepping and hasattr(model, 'evaluate_momentum_tendency_with_diagnostics'):
            ab2_tendency, diagnostics = model.evaluate_momentum_tendency_with_diagnostics(
                state, inputs=inputs)
            return ab2_tendency, None, diagnostics
        if hasattr(model, 'evaluate_slow_tendency_split'):
            return model.evaluate_slow_tendency_split(
                state, assembly_policy=assembly_policy, inputs=inputs,
                tracer_ab2_names=tuple(name for name, enabled in (
                    ("temp", self.ab2_on_temperature_tendency),
                    ("salt", self.ab2_on_salinity_tendency),
                ) if enabled and self.ab_eps > 0.0))
        ab2_tendency, diagnostics = model.evaluate_slow_tendency_with_diagnostics(
            state, inputs=inputs)
        return ab2_tendency, None, diagnostics

    def _publish_boundary(
        self, key: tuple, payload: "BoundaryPayload"
    ) -> "tuple[BoundaryPayload, KppStepForcing | None]":
        """Stamp the prepare key and retain the payload for the idempotent path.

        The slot is prepare-owned: ``build_step_plan`` reads the return value,
        and the traced step reads ``StepPlan.boundary``.  Nothing in the step
        body reads this attribute.
        """
        object.__setattr__(self, "_prepared_step_key", key)
        object.__setattr__(self, "_prepared_boundary", payload)
        return payload, self._kpp_forcing_inputs

    def _prepare_step_entry(self, *, model, plan):
        implicit_vertical_advection_active = (
            self.temperature_implicit_vertical_advection
            or self.salinity_implicit_vertical_advection
            or self.momentum_implicit_vertical_advection
        )
        temperature_advection_scheme = salinity_advection_scheme = None
        multi_dim_advection = False
        if self.temperature_implicit_vertical_advection or self.salinity_implicit_vertical_advection:
            _, _, temp_vertical, salt_vertical = admission._runtime_tracer_schemes(model.dynamics_modules)
            temperature_advection_scheme = admission._tracer_scheme(temp_vertical)
            salinity_advection_scheme = admission._tracer_scheme(salt_vertical)
            if ((self.temperature_implicit_vertical_advection
                 and temperature_advection_scheme is TracerFluxScheme.DST3_FL)
                    or (self.salinity_implicit_vertical_advection
                        and salinity_advection_scheme is TracerFluxScheme.DST3_FL)):
                multi_dim_advection = model.tracer_multi_dim_advection
        if implicit_vertical_advection_active:
            # The counter validation is
            # host work (``prepare_step_host`` runs ``_validate_step_index_state``
            # before the step body is entered).
            admission._require_implicit_vertical_advection_execution_protocol(
                self._implicit_vertical_advection_execution(model),
            )
        bundle = coerce_time_step_bundle(plan.times.bundle)
        dt_mom = bundle.dt_mom
        dt_tracer = bundle.dt_tracer
        dt_freesurf = bundle.dt_freesurf
        grid = model.grid  # type: ignore[attr-defined]
        # One-shot: migrate the grid's large constant masks / face_geometry onto
        # the compute device so the cube operators stop re-transferring them
        # (~285 MB hFac each) host→device every step.  Idempotent + gated by
        # DEFGCM_DEVICE_RESIDENT_MASKS (default ON); no-op after the first step and
        # on the regional path when the grid is already on-device.
        alpha = self.implicit_surf_press  # α
        beta = self.implicit_div_2d_flow  # β
        # Shared predictor/correction segments apply pfFacMom to surface PG;
        # the free-surface solve retains alpha. Regional pressure modules read
        # the same switch through EvaluationInputs and retain their diagnostics.
        return (implicit_vertical_advection_active, bundle, dt_mom, dt_tracer,
                dt_freesurf, grid, alpha, beta, temperature_advection_scheme,
                salinity_advection_scheme, multi_dim_advection)

    def _prepare_regional_entry(
        self, *, state, model, grid, plan, dt_tracer, dt_freesurf,
        hydro_history, boundary_evaluation,
    ):
        if ckpt_dump.ENABLED:
            record_checkpoint(
                "ic",
                eta=state.eta.data,
                u=state.u.data,
                v=state.v.data,
                temp=state.temp.data,
                salt=state.salt.data,
            )

        # JMD95P INI_PRESSURE-equivalent seed: ``_seed_total_phi_hyd_picard``
        # has filled ``self._cached_total_phi_hyd`` before the first step, so
        # the first EOS evaluation uses true MIT-aligned dynamic pressure
        # rather than silently falling back to JMD95Z (see d1a-only ablation
        # 49-317× degradation and the seed's docstring).
        _dist_trace("step begin")
        # Initial velocities and UPDATE_ETAH use startTime samples. The first
        # geometry calculation uses the initial eta field itself (myIter=-1);
        # future samples belong to the later boundary application events.
        boundary = plan.boundary
        if boundary.current.sampled:
            boundary.current.install()
        else:
            raise RuntimeError("regional step requires declared entry samples")
        initial_surface_geometry = (
            self._surface_dr_active(grid) and self._surface_geometry.surface is None
        )
        initial_r_star = self._r_star_active(grid) and self._r_star.factors is None
        # A fresh run's first step, which reads the state INITIALISE_VARIA
        # leaves (PmEpR = 0).  Taken before the etaH initialisation below
        # writes the key.
        first_step = hydro_history is None and self._continuity.eta_h is None
        if self._surface_dr_active(grid) and (
            boundary.current.obc_eta is None or boundary.future.obc_eta is None
        ):
            raise RuntimeError("nonlinear geometry requires current and application ETA samples")
        # ── Nonlinear free surface: surface-level hFac time layers ──
        # MITgcm keeps TWO surface thicknesses in flight and they belong to
        # different free-surface time levels (forward_step.F, pruned to this
        # configuration -- staggerTimeStep=F, select_rStar=0, doResetHFactors=F):
        #
        #   entering the step   _hFac* = h(eta^{n-1})   hFac_surf* = h(eta^n)
        #   :732/:791           THERMODYNAMICS / DYNAMICS run on _hFac*
        #   :851                UPDATE_SURF_DR:  _hFac* := hFac_surf* = h(eta^n)
        #   :903/:927           SOLVE_FOR_PRESSURE / INTEGR_CONTINUITY -> eta^{n+1}
        #   :958                CALC_SURF_DR:    hFac_surf* := h(eta^{n+1})
        #
        # So the geometry the tendencies see lags the free surface by one step,
        # and every hFac/hFac_surf rescale factor is h(eta^{n-1})/h(eta^n) --
        # Refreshing geometry here would make both operands h(eta^n) and
        # incorrectly turn the rescale into a no-op.
        #
        # This block therefore only *computes* hFac_surf; ``_apply_surf_dr``
        # below writes it into the geometry at the MITgcm :851 point.
        if initial_r_star:
            self._initialise_r_star(model, grid, state, dt_freesurf=dt_freesurf)
        if self._surface_dr_active(grid):
            if self._surface_geometry.resting is None:
                object.__setattr__(self, "_surface_geometry", ifs_geometry.SurfaceGeometry(
                    resting=(grid.masks.hFacC.clone(), grid.masks.hFacW.clone(), grid.masks.hFacS.clone()),
                ))
                # The shallow-column precondition (a host-side ``.any()``) runs
                # once in ``prepare_step_host`` , not in the body.
            if self._surface_geometry.surface is None:
                # INITIALISE_VARIA:314-321 installs the initial etaN geometry.
                # After the continuity seed below, :343-345 refreshes only
                # hFac_surf from etaH, retaining this grid hFac for physics.
                #
                # Whole-model checkpoints restore both geometry time levels
                # before prepare, so a continuation does not take this branch.
                _eta_seed = self._continuity.eta_h
                if _eta_seed is None:
                    _eta_seed = state.eta.data
                # OBCS_APPLY_SURF_DR:60-63 sets useOBeta=false for myIter=-1.
                # Reuse the field-form override to take all four sides from
                # initial eta, before prescribed ETA enters the current layer;
                # the open-boundary faces then depend on initial eta, as there.
                object.__setattr__(self, "_surface_geometry", self._refresh_hfac_surf(
                    grid, _eta_seed, obc_eta=_eta_seed,
                ))
                self._apply_surf_dr(grid)

        # MITgcm flow: OBCS_CALC(myTime+dt) precomputes future OBC values
        # (do_oceanic_phys.F:320), but the tendency evaluation at line 730 uses
        # CURRENT state values directly — OBCS_APPLY_UV is only called AFTER
        # TIMESTEP+IMPLDIFF (dynamics.F:610).  So the boundary faces enter the
        # tendency window on the CURRENT clock and move to futureTime for the
        # post-correction enforces (line 835 and 969).
        #
        _dist_trace("initial enforce_boundaries begin")
        # OBCS_CALC_STEVENS diagnoses the future boundary tendency from the
        # step-start state. Only THERMODYNAMICS advances the boundary tracers.
        # The phases also carry this step's Flather targets, evaluated here
        # from the step-start state under the current samples; the predicted,
        # corrected and final applications reuse them.
        enforce_velocity, enforce_boundary_tracers = tracer_boundary_phases(
            model, state, dt_tracer, evaluation=boundary_evaluation)
        bounded = enforce_velocity(
            state, record_budget=False, initial=initial_surface_geometry, step_entry=True,
        )
        _dist_trace("initial enforce_boundaries done")
        eta_h = None
        if hydro_history is None:
            eta_h = self._continuity.eta_h
            if eta_h is None:
                eta_h = self._initial_eta_h(bounded, grid, dt_freesurf)
        if initial_r_star:
            # INITIALISE_VARIA's INTEGR_CONTINUITY on the bounded initial flow
            # (initialise_varia.F:332-335, the myIter = nIter0 branch
            # dEtaHdt = -hDivFlow*recip_rA of integr_continuity.F:165-172).
            u0 = bounded.u.data
            grid.masks.set_r_star_thickness_rate(r_star.r_star_thickness_rate(
                ifs_freshwater.regional_continuity_rate(
                    u_final=u0, v_final=bounded.v.data,
                    layer_thickness=grid.layer_thickness.to(
                        dtype=u0.dtype, device=u0.device).view(1, grid.nz, 1, 1),
                    hFacW=grid.masks.hFacW.to(dtype=u0.dtype, device=u0.device),
                    hFacS=grid.masks.hFacS.to(dtype=u0.dtype, device=u0.device),
                    grid=grid,
                ),
                self._r_star.columns.recip_depth_c,
            ))
        if initial_surface_geometry:
            eta_h = apply_surface_eta_boundary(
                eta_h, grid, obc_eta=boundary.current.obc_eta,
            )
            # Preserve the old/current pair until UPDATE_SURF_DR, after the
            # tendencies have used the old grid geometry and surface ratio.
            object.__setattr__(self, "_surface_geometry", self._refresh_hfac_surf(
                grid, eta_h, obc_eta=boundary.current.obc_eta,
            ))
        if ckpt_dump.ENABLED:
            record_checkpoint(
                "bounded",
                eta=bounded.eta.data,
                u=bounded.u.data,
                v=bounded.v.data,
                temp=bounded.temp.data,
                salt=bounded.salt.data,
            )
        return boundary, enforce_velocity, enforce_boundary_tracers, bounded, eta_h, first_step

    def _finalize_regional_state(
        self, *, state, model, grid, corrected_bounded, u_final, v_final,
        eta_final, temp_new, salt_new, enforce_velocity, enforce_boundary_tracers,
        internal_nh_active, phi_nh_history, gw_nm1_history,
        implicit_vertical_advection_active, tracer_w, hydro_history, eta_h_history,
    ):
        if internal_nh_active:
            _dist_trace("internal nonhydrostatic cache refresh begin")
            # Refresh internal NH cache after final tracer/boundary state is known.
            # Producer is LOCOMgcm downward-positive interface w [B,nz+1,ny,nx];
            # next NH-A consumes MIT upward-positive Nr-sized wVel.
            from defgcm.dynamics.vertical_velocity import diagnose_vertical_velocity

            preview_state = corrected_bounded.with_tensors(
                eta=eta_final,
                u=u_final,
                v=v_final,
                temp=temp_new,
                salt=salt_new,
            )
            _, w_iface_preview, _ = diagnose_vertical_velocity(preview_state, model.grid)
            w_vel_history = _carry(
                ifs_nh.mit_w_velocity_at_levels(w_iface_preview, nz=grid.nz)
            )
            object.__setattr__(self, "_nh_history", ifs_nh.NHHistory(
                phi_nh_history, gw_nm1_history, w_vel_history, True, True, True,
            ))
            _dist_trace("internal nonhydrostatic cache refresh done")

        # 10. Build final state with etaN and apply boundaries
        #     state.eta = etaN (used for explicit PG and CG initial guess)
        updated = state.with_tensors(
            eta=eta_final,
            u=u_final,
            v=v_final,
            temp=temp_new,
            salt=salt_new,
        )
        apply_final = enforce_boundary_tracers if self.stagger_tracer_stepping else enforce_velocity
        result, result_u, result_v = ifs_publication.apply_boundary_and_sync(
            updated, grid, apply_boundary=apply_final,
            boundary_begin="final enforce_boundaries begin",
            boundary_done="final enforce_boundaries done",
            sync_begin="final shared velocity sync begin",
            sync_done="final shared velocity sync done",
        )
        if result_u is not result.u.data or result_v is not result.v.data:
            result = result.with_tensors(u=result_u, v=result_v)
        # TRACERS_CORRECTION_STEP: convective adjustment of the updated tracers
        # (tracers_correction_step.F:114-118).  It follows THERMODYNAMICS'
        # OBCS_APPLY_TS (thermodynamics.F:359, forward_step.F:1025), so the
        # open-boundary points keep the adjusted values.
        adjustment = convective_adjustment_of(model.dynamics_modules)
        if adjustment is not None:
            salt_to_adjust = result.salt.data
            if self.check_salinity_for_negative_values and grid.nz > 1:
                # convective_adjustment.F:123-136 resets layers k-1 and k at
                # interface k before mixing them; mixing non-negative values keeps
                # them non-negative, so resetting the column first is the same.
                salt_to_adjust = zero_negative_salinity(salt_to_adjust)
            temp_adjusted, salt_adjusted = adjustment.adjust(
                result.temp.data, salt_to_adjust, grid=grid, modules=model.dynamics_modules,
            )
            result = result.with_tensors(temp=temp_adjusted, salt=salt_adjusted)
        if ckpt_dump.ENABLED:
            record_checkpoint("tracer", temp=result.temp.data, salt=result.salt.data)

        # Cache w_interface from corrected state for next step's vertical
        # shear.  MITgcm computes wVel via INTEGR_CONTINUITY AFTER
        # MOMENTUM_CORRECTION_STEP (forward_step.F:914 and :928 at MITgcm 27256b2d), so the vertical
        # shear in the NEXT timestep uses w from the corrected velocities
        # of the CURRENT timestep.
        #
        # CONVENTION: diagnose_vertical_velocity returns w in LOCOMgcm's
        # convention (downward-positive, docs/NUMERICS.md sign
        # conventions), but the vector-invariant vertical-shear kernels are
        # MITgcm-aligned (mom_vi_u_vertshear.F) and expect upward-positive
        # w.  Negate at the producer-consumer boundary so the cache stores
        # MITgcm-convention w.  Alignment harnesses that inject MIT wVel
        # directly (already upward-positive) use the same cache key, so this
        # convention is consistent.
        _dist_trace("final vertical velocity cache begin")
        w_next_mit = None
        if implicit_vertical_advection_active:
            if tracer_w is None:
                raise RuntimeError(
                    "implicit vertical advection promotion lacks tracer-time wrapper"
                )
            # The one stored lagged w; the next step's momentum solve reads it
            # too (``momentum_w`` above).
            if hydro_history is None:
                object.__setattr__(self, "_continuity", ContinuityHistory(
                    eta_h_history, _carry(tracer_w.data),
                ))
        else:
            w_next_mit = prepare.diagnose_mit_interface_velocity(state=result, grid=model.grid)
            if hydro_history is None:
                object.__setattr__(self, "_continuity", ContinuityHistory(
                    eta_h_history, _carry(w_next_mit),
                ))
        # advance_step_index increments the counter in the eager epilogue.
        _dist_trace("final vertical velocity cache done")
        return result, w_next_mit

    def _publish_step_outcome(
        self, *, result, ab_history, hydro_history, next_ab_history,
        eta_H_new, w_next_mit, cg2d_iters, cg2d_residual, eta_n, dphi_nh,
        nh_iters, nh_first_residual, nh_last_residual,
    ) -> StepOutcome:
        # Opt-in only: releasing cached segments forces allocations on the
        # next step. The allocator already frees and retries after a failed
        # allocation; request this only when persistent fragmentation requires it.
        import os as _os_ec
        if torch.cuda.is_available() and _os_ec.environ.get(
            "DEFGCM_EMPTY_CACHE"
        ) == "1":
            torch.cuda.empty_cache()

        if ckpt_dump.ENABLED:
            record_checkpoint(
                "final",
                eta=result.eta.data,
                u=result.u.data,
                v=result.v.data,
                temp=result.temp.data,
                salt=result.salt.data,
            )
        _dist_trace("step done")
        retained = EMPTY_MAPPING
        if ab_history is not None:
            if hydro_history is not None:
                from defgcm.step.carry import HydroHistory

                hydro_history = HydroHistory(eta_H_new, w_next_mit)
            retained = {"ab_history": next_ab_history, "hydro_history": hydro_history}
        return StepOutcome(
            state=result,
            diagnostics=StepDiagnostics(
                cg2d_iters=cg2d_iters,
                cg2d_residual=cg2d_residual,
                eta_n=eta_n,
                dphi_nh=dphi_nh,
                cg3d_iters=nh_iters,
                cg3d_first_residual=nh_first_residual,
                cg3d_residual=nh_last_residual,
            ),
            retained=retained,
        )

    def _prepare_step_forcing(
        self, *, bounded, model, grid, plan, dt_tracer, first_step,
    ):
        seaice = self._package_modules.seaice
        if seaice is not None:
            # EXF, THSICE_MAIN and SEAICE_MODEL read the step-entry ocean before
            # FREEZE_SURFACE (do_oceanic_phys.F:376-474,557).
            object.__setattr__(self, "_seaice", seaice.advance(
                self._seaice, bounded, grid, dt=dt_tracer))
        freezing_flux = None
        if self.allow_freezing:
            freezing_point = self.freezing_point
            previous_surface, surface = ifs_freshwater.freeze_surface(
                bounded.temp.data, freezing_point=freezing_point,
            )
            if self.kpp_enabled:
                dtype, device = surface.dtype, surface.device
                freezing_flux = ifs_freshwater.surface_freezing_flux(
                    previous_surface, freezing_point=freezing_point, dt_tracer=dt_tracer,
                    layer_thickness=grid.layer_thickness_for(dtype, device),
                    hfac_c=grid.hfac_c_for(dtype, device),
                )
            bounded = ifs_freshwater.install_frozen_surface(bounded, surface=surface)
        if self.check_salinity_for_negative_values:
            # The step's first FIND_RHO_2D, rhoInSitu (do_oceanic_phys.F:758-765),
            # follows FREEZE_SURFACE and EXTERNAL_FORCING_SURF and resets the
            # prognostic salt that THERMODYNAMICS and the column physics read.
            bounded = bounded.with_tensors(salt=zero_negative_salinity(bounded.salt.data))

        if self.stream_trace.hook is not None:
            from defgcm.dynamics.vertical_velocity import diagnose_vertical_velocity

            _, bounded_w, _ = diagnose_vertical_velocity(bounded, grid)
            self.stream_trace.emit_stream_trace(
                "step.post_oceanic_physics",
                u=bounded.u.data,
                v=bounded.v.data,
                w=bounded_w,
                temp=bounded.temp.data,
                salt=bounded.salt.data,
            )

        empmr_mass_flux = None
        mass_to_r_unit = 1.0
        if self.use_real_freshwater_flux:
            empmr_mass_flux, mass_to_r_unit = ifs_surface.real_freshwater_mass_flux(
                getattr(model, "dynamics_modules", ()) or (),
                dtype=bounded.eta.data.dtype,
                device=bounded.eta.data.device,
            )
        pmep_r = self._pmep_r_for_step(
            grid, bounded.salt.data.dtype, bounded.salt.data.device, first_step=first_step,
            empmr_mass_flux=empmr_mass_flux,
        )

        # Collect current producer fields before the KPP diagnose; fixed host
        # inputs supply fields without a producer in this composition.
        # GGL90 reads the same surfaceForcingU/V (do_oceanic_phys.F:579,
        # ggl90_calc.F:766-797); KPP alone reads the returned bundle downstream.
        kpp_forcing = None
        if self.reads_kpp_forcing:
            kpp_forcing = self._populate_kpp_forcing_from_model(
                model, bounded, grid, bounded.temp.data.dtype, bounded.temp.data.device,
                forcing_inputs=plan.kpp_forcing_inputs,
                pmep_r=pmep_r,
            )

        # KPP vertical mixing: diagnose diffusivities from the step-start state
        # (pure diagnostic).  Consumed by the implicit vertical viscosity
        # (momentum) and the tracer vertical diffusion + ghat below.
        kpp_inputs = (
            self._prepare_regional_kpp_inputs(
                bounded, model, grid, bounded.temp.data.dtype, bounded.temp.data.device,
                forcing=kpp_forcing, freezing_flux=freezing_flux,
            )
            if getattr(self, "kpp_enabled", False)
            else None
        )
        kpp_fields = (
            kpp_inputs.diagnose(
                bounded.temp.data, bounded.salt.data,
                bounded.u.data, bounded.v.data, grid=grid,
            )
            if kpp_inputs is not None else None
        )
        if ckpt_dump.ENABLED:
            _kpp_ckpt = kpp_fields
            _kpp_tensors = (
                {}
                if _kpp_ckpt is None
                else {
                    "kappa_ru": _kpp_ckpt.kappa_ru,
                    "kappa_rv": _kpp_ckpt.kappa_rv,
                    "diff_kzt": _kpp_ckpt.diff_kzt,
                    "diff_kzs": _kpp_ckpt.diff_kzs,
                    "ghat": _kpp_ckpt.ghat,
                    "hbl": _kpp_ckpt.hbl,
                }
            )
            record_checkpoint("kpp", **_kpp_tensors)
        kpp = KppEvaluation(kpp_inputs, kpp_fields) if kpp_inputs is not None else None
        tracer_kr = self.tracer_vertical_diffusivity_kr
        scheme = self.vertical_mixing_scheme
        ivdc_active = self.ivdc_kappa != 0.0
        if ivdc_active or scheme != "none":
            # One kRef = k density pair per step, shared by ivdc_kappa and the
            # scheme (do_oceanic_phys.F:803-887).
            rho_in_situ, rho_km1 = self._mixing_densities(bounded, model, grid)
            mask_c, recip_drc = columns.mixing_column_geometry(
                grid, dtype=rho_in_situ.dtype, device=rho_in_situ.device,
            )
        if scheme != "none":
            # the background columns viscArNr, diffKrNrS of the scheme kernels
            dtype, device = rho_in_situ.dtype, rho_in_situ.device
            visc_arnr = torch.full((grid.nz,), float(self.implicit_viscosity), dtype=dtype, device=device)
            diffkr_nr_s = torch.full((grid.nz,), float(self.tracer_vertical_diffusivity_kr),
                                     dtype=dtype, device=device)
        if ivdc_active or scheme == "ggl90" or scheme == "kl10":
            # grad_sigma.F:84-98, from the same pair
            sigma_r = vertical_density_gradient(rho_in_situ, rho_km1, mask_c=mask_c, recip_drc=recip_drc)
        if ivdc_active:
            # do_oceanic_phys.F:803-887
            tracer_kr = columns.ivdc_tracer_diffusivity_at_step(
                sigma_r=sigma_r, ivdc_kappa=self.ivdc_kappa, background_kr=tracer_kr,
            )
        contributions = (kpp_fields,) if self.kpp_enabled else ()
        if scheme == "pp81" or scheme == "my82":
            # PP81_CALC / MY82_CALC after KPP (do_oceanic_phys.F:967-998)
            contributions = (columns.centre_scheme_contribution_at_step(
                scheme=scheme, parameters=self.vertical_mixing_parameters, grid=grid,
                rho_in_situ=rho_in_situ, rho_km1=rho_km1, u=bounded.u.data, v=bounded.v.data,
                mask_c=mask_c, recip_drc=recip_drc, gravity=self.gravity,
                mass2r_unit=self._prepared_mass_to_r_unit,
                background_ar=float(self.implicit_viscosity),
                visc_arnr=visc_arnr, diffkr_nr_s=diffkr_nr_s,
            ),)
        elif scheme == "kl10":
            # KL10_CALC after PP81 (do_oceanic_phys.F:978-986)
            contributions = (columns.kl10_contribution_at_step(
                visc_max=self.vertical_mixing_parameters, grid=grid, rho_in_situ=rho_in_situ,
                sigma_r=sigma_r, mask_c=mask_c, recip_drc=recip_drc, gravity=self.gravity,
                mass2r_unit=self._prepared_mass_to_r_unit,
                visc_arnr=visc_arnr, diffkr_nr_s=diffkr_nr_s,
            ),)
        elif scheme == "ggl90":
            # GGL90_CALC (do_oceanic_phys.F:1000-1014) steps last step's TKE
            if kpp_forcing.surf_forc_u is None or kpp_forcing.surf_forc_v is None:
                raise ValueError("GGL90 enabled but surf_forc_u/surf_forc_v not set")
            tke, ggl90_contribution = columns.ggl90_step(
                parameters=self.vertical_mixing_parameters, grid=grid, tke=self._ggl90_tke,
                sigma_r=sigma_r, u=bounded.u.data, v=bounded.v.data,
                surf_forc_u=kpp_forcing.surf_forc_u, surf_forc_v=kpp_forcing.surf_forc_v,
                mask_c=mask_c, recip_drc=recip_drc, gravity=self.gravity,
                recip_rho_const=self._prepared_mass_to_r_unit, dt=dt_tracer,
                visc_arnr=visc_arnr, diffkr_nr_s=diffkr_nr_s,
            )
            object.__setattr__(self, "_ggl90_tke", tke)
            contributions = (ggl90_contribution,)
        kappa_ru, kappa_rv = columns.assemble_vertical_viscosity(
            background_ar=float(self.implicit_viscosity), contributions=contributions,
        )
        mixing = columns.VerticalMixingStep(
            kappa_ru=kappa_ru, kappa_rv=kappa_rv, tracer_kr=tracer_kr, contributions=contributions,
        )
        return bounded, freezing_flux, empmr_mass_flux, mass_to_r_unit, pmep_r, kpp_forcing, kpp, mixing

    def _prepare_step_tendency_inputs(
        self, *, bounded, model, grid, hydro_history, pmep_r, kpp_forcing, kpp, mixing,
    ):
        kappa_ru_floor = kappa_rv_floor = None
        if self.vertical_mixing_scheme != "none":
            # calc_viscosity.F:121-130: the schemes copy interface Nr to Nr+1,
            # which the no-slip bottom drag of the same step reads.
            kappa_ru_floor = bottom_interface_viscosity(mixing.kappa_ru)[..., -1:, :, :]
            kappa_rv_floor = bottom_interface_viscosity(mixing.kappa_rv)[..., -1:, :, :]
        coriolis_u = coriolis_v = None
        if self.vertical_mixing_scheme == "ggl90" and self.vertical_mixing_parameters.use_langmuir:
            # GGL90_ADD_STOKESDRIFT of this step's surfaceForcingU/V
            coriolis_u, coriolis_v = columns.ggl90_coriolis_velocity(
                self.vertical_mixing_parameters, grid, u=bounded.u.data, v=bounded.v.data,
                surf_forc_u=kpp_forcing.surf_forc_u, surf_forc_v=kpp_forcing.surf_forc_v)
        evaluation_inputs = self._evaluation_inputs(model, bounded, grid, hydro_history,
                                                    pmep_r=pmep_r, kappa_ru_floor=kappa_ru_floor,
                                                    kappa_rv_floor=kappa_rv_floor,
                                                    coriolis_u=coriolis_u, coriolis_v=coriolis_v)
        momentum_w = None
        if self.momentum_implicit_vertical_advection:
            momentum_w = ifs_tendency.momentum_w_from_evaluation(evaluation_inputs)
        resolved_3d_coriolis = self._resolved_3d_coriolis

        # 1. Evaluate slow tendencies.
        #    When stagger_tracer_stepping=True, only evaluate momentum-phase
        #    modules now (tracer modules evaluated after velocity correction).
        #    This matches MITgcm staggerTimeStep where THERMODYNAMICS runs
        #    after SOLVE_FOR_PRESSURE and MOMENTUM_CORRECTION.
        assembly_policy = self._resolved_assembly_policy or AssemblyPolicy.regional(self)
        # Only fields actually extrapolated move their nonlocal source. Euler
        # fields retain the old state-plus-source rounding in mix_kpp_tracers.
        kpp_ab_names = tuple(
            name for name, enabled in (
                ("temp", self.ab2_on_temperature_tendency),
                ("salt", self.ab2_on_salinity_tendency),
            ) if kpp is not None and self.ab_eps > 0.0 and enabled
        )
        kpp_post_names = tuple(name for name in ("temp", "salt") if name not in kpp_ab_names)
        # timestep.F:275-325 rescales flux-form momentum before adding the
        # staggered HPG. Keep forcing/dissipation in the rescaled buckets.
        # The CD scheme takes the staggered HPG apart from gUtmp
        # (cd_code_scheme.F:77-84), so it defers the HPG as well.
        defer_hpg = (
            self.stagger_tracer_stepping
            and self.momentum_pressure_forcing
            and (
                (self.nonlin_free_surf >= 2
                 and self._nonlin_free_surf_active(grid)
                 and not self._resolved_vector_invariant_momentum)
                or cd_scheme_of(model.dynamics_modules) is not None
            )
        )
        momentum_inputs = (
            dataclasses.replace(evaluation_inputs, momentum_pressure_forcing=False)
            if defer_hpg else evaluation_inputs
        )
        _dist_trace("evaluate tendency begin")
        return (evaluation_inputs, momentum_w, resolved_3d_coriolis, assembly_policy,
                kpp_ab_names, kpp_post_names, defer_hpg, momentum_inputs)

    def _complete_step_tendencies(
        self, *, bounded, model, grid, hydro_history, kpp, kpp_fields, dt_mom,
        ab2_tendency, diagnostics, kpp_ab_names,
    ):
        _dist_trace("evaluate tendency done")
        kpp_nonlocal = None
        if not self.stagger_tracer_stepping and kpp_ab_names:
            temp_kz, salt_kz = self._kpp_nonlocal_diffusivities(
                kpp_fields,
                self._gm_kwz_contributions(
                    diagnostics, grid, dtype=kpp_fields.diff_kzt.dtype,
                    device=kpp_fields.diff_kzt.device,
                ) if self.kpp_ghat_use_total_diffus else (),
            )
            ab2_tendency, kpp_nonlocal = ifs_tendency._add_kpp_nonlocal_to_tracer_bucket(
                ab2_tendency, state=bounded, kpp=kpp_fields,
                temp_diffusivity=temp_kz, salt_diffusivity=salt_kz,
                layer_thickness=kpp.inputs.static.layer_thickness,
                grid=grid, tracer_names=kpp_ab_names,
            )
        if ckpt_dump.ENABLED:
            record_checkpoint(
                "tendency",
                u=ab2_tendency.u,
                v=ab2_tendency.v,
                temp=ab2_tendency.temp,
                salt=ab2_tendency.salt,
            )
        # totPhiHyd is updated by CALC_PHI_HYD inside DYNAMICS (dynamics.F:482,
        # diags_phi_hyd.F:62), which momStepping=.FALSE. skips: the history then
        # keeps its INI_PRESSURE value (ini_pressure.F:112), seeded before the
        # first step by _seed_total_phi_hyd_picard.
        if hydro_history is None and self.mom_stepping:
            self._cache_total_phi_hyd_from_diagnostics(diagnostics)
        model.accumulate_tendency_diagnostics(diagnostics, dt_mom)
        return ab2_tendency, kpp_nonlocal

    def _extrapolate_step_ab(self, *, ab2_tendency, ab_history):
        # 1b. Adams-Bashforth 2nd order extrapolation
        #     MITgcm: G* = G^n + (0.5+ε)*(G^n - G^{n-1})
        #     gad_init_fixed.F:143-163 derives temperature and salinity AB
        #     independently from their stepping flags and advection schemes.
        #     adams_bashforth2.F: first step uses forward Euler (abFac=0).
        next_ab_history = ab_history
        if self.ab_eps > 0.0 and ab_history is not None:
            ab_fac = ifs_tendency.ab2_factor(self.ab_eps)
            tend_u = ifs_tendency.extrapolate_valid_ab2(
                valid=ab_history["mom_valid"], current=ab2_tendency.u,
                previous=ab_history["mom_u"], factor=ab_fac,
            )
            tend_v = ifs_tendency.extrapolate_valid_ab2(
                valid=ab_history["mom_valid"], current=ab2_tendency.v,
                previous=ab_history["mom_v"], factor=ab_fac,
            )
            tend_temp = ab2_tendency.temp
            tend_salt = ab2_tendency.salt
            if self.ab2_on_temperature_tendency:
                tend_temp = ifs_tendency.extrapolate_valid_ab2(
                    valid=ab_history["temp_valid"], current=ab2_tendency.temp,
                    previous=ab_history["temp"], factor=ab_fac,
                )
            if self.ab2_on_salinity_tendency:
                tend_salt = ifs_tendency.extrapolate_valid_ab2(
                    valid=ab_history["salt_valid"], current=ab2_tendency.salt,
                    previous=ab_history["salt"], factor=ab_fac,
                )
        elif self.ab_eps > 0.0 and self._momentum_ab is not None:
            prev_tend = self._momentum_ab
            ab_fac = ifs_tendency.ab2_factor(self.ab_eps)
            tend_u = ifs_tendency.extrapolate_ab2(ab2_tendency.u, prev_tend.u, factor=ab_fac)
            tend_v = ifs_tendency.extrapolate_ab2(ab2_tendency.v, prev_tend.v, factor=ab_fac)
            # A seeded history (``regional_cold_start``) holds zeros and false
            # flags: ``where`` keeps the first step's forward-Euler tendency
            # (adams_bashforth2.F abFac=0) in the graph every step uses.
            # Histories from checkpoints older than the flags carry none.
            mom_valid = prev_tend.mom_valid
            if mom_valid is not None:
                tend_u = ifs_tendency.select_valid_ab2(
                    valid=mom_valid, extrapolated=tend_u, current=ab2_tendency.u,
                )
                tend_v = ifs_tendency.select_valid_ab2(
                    valid=mom_valid, extrapolated=tend_v, current=ab2_tendency.v,
                )
            if self.ab2_on_temperature_tendency and prev_tend.temp is not None:
                tend_temp = ifs_tendency.extrapolate_ab2(ab2_tendency.temp, prev_tend.temp, factor=ab_fac)
                temp_valid = prev_tend.temp_valid
                if temp_valid is not None:
                    tend_temp = ifs_tendency.select_valid_ab2(
                        valid=temp_valid, extrapolated=tend_temp, current=ab2_tendency.temp,
                    )
            else:
                tend_temp = ab2_tendency.temp
            if self.ab2_on_salinity_tendency and prev_tend.salt is not None:
                tend_salt = ifs_tendency.extrapolate_ab2(ab2_tendency.salt, prev_tend.salt, factor=ab_fac)
                salt_valid = prev_tend.salt_valid
                if salt_valid is not None:
                    tend_salt = ifs_tendency.select_valid_ab2(
                        valid=salt_valid, extrapolated=tend_salt, current=ab2_tendency.salt,
                    )
            else:
                tend_salt = ab2_tendency.salt
        else:
            tend_u = ab2_tendency.u
            tend_v = ab2_tendency.v
            tend_temp = ab2_tendency.temp
            tend_salt = ab2_tendency.salt

        if ckpt_dump.ENABLED:
            record_checkpoint(
                "ab2", u=tend_u, v=tend_v, temp=tend_temp, salt=tend_salt
            )

        self.stream_trace.emit_stream_trace(
            "momentum.post_ab",
            g_u=tend_u,
            g_v=tend_v,
        )

        # Save current tendency for next step's AB2
        if self.ab_eps > 0.0:
            history_u, history_v = ifs_tendency.retain_momentum_ab(
                ab2_tendency.u, ab2_tendency.v,
            )
            cache_dict = {"u": history_u, "v": history_v}
            if self.ab2_on_temperature_tendency:
                cache_dict["temp"] = (
                    ifs_tracer.retain_tracer_ab(ab2_tendency.temp)
                    if ab_history is None else ifs_tendency.clone_tracer_ab(ab2_tendency.temp)
                )
            if self.ab2_on_salinity_tendency:
                cache_dict["salt"] = (
                    ifs_tracer.retain_tracer_ab(ab2_tendency.salt)
                    if ab_history is None else ifs_tendency.clone_tracer_ab(ab2_tendency.salt)
                )
            if ab_history is None:
                object.__setattr__(self, "_momentum_ab", ifs_tendency.assemble_momentum_ab_history(
                    cache_dict=cache_dict, device=ab2_tendency.u.device,
                ))
            else:
                next_ab_history = ifs_tendency.assemble_carry_momentum_ab_history(
                    u=cache_dict["u"], v=cache_dict["v"], mom_valid=ab_history["mom_valid"],
                )
                for name in ("temp", "salt"):
                    if name in ab_history:
                        next_ab_history.update(ifs_tendency.assemble_carry_tracer_ab_history(
                            name=name, tendency=cache_dict[name], valid=ab_history[name + "_valid"],
                        ))
        return tend_u, tend_v, tend_temp, tend_salt, next_ab_history

    def _prepare_momentum_prediction(
        self, *, bounded, model, grid, tend_u, tend_v, tend_temp, tend_salt,
        post_ab2_tendency, defer_hpg, diagnostics, evaluation_inputs, dt_mom, ab2_tendency,
    ):
        if post_ab2_tendency is not None:
            tend_u, tend_v, tend_temp, tend_salt = ifs_tendency.add_post_ab2(
                tend_u, tend_v, tend_temp, tend_salt, post_ab2_tendency)
        pressure_tendencies = []
        if defer_hpg:
            hpg_diagnostics = dataclasses.replace(diagnostics, inputs=evaluation_inputs)
            for module in model.slow_momentum_modules:
                if module.momentum_term_class == MomentumTermClass.PRESSURE_GRADIENT:
                    pressure_tendencies.append(model.boundary_condition.enforce_tendency(
                        module.evaluate(bounded, hpg_diagnostics, grid), grid,
                    ))
        # MITgcm timestep.F:228-270: the CD scheme steps the D-grid velocity
        # from gUtmp (post AB, with the terms kept outside AB) and adds its
        # Coriolis terms outside AB, before the nonlinear free-surface rescale.
        cd_scheme = cd_scheme_of(model.dynamics_modules)
        if cd_scheme is not None:
            cd_g_u, cd_g_v = tend_u, tend_v
            if cd_scheme.steps_with_pre_ab_tendency:
                # timestep.F:148-158, 209-226: gU^n before AB, then the terms
                # kept outside AB.
                cd_g_u, cd_g_v = ab2_tendency.u, ab2_tendency.v
                if post_ab2_tendency is not None:
                    cd_g_u = cd_g_u + post_ab2_tendency.u
                    cd_g_v = cd_g_v + post_ab2_tendency.v
            # cd_code_scheme.F:77-84: phxFac = phyFac = pfFacMom when staggered,
            # zero otherwise; admit_cd_scheme leaves exactly one HPG module then.
            hydrostatic_gradient = None
            if self.stagger_tracer_stepping and self.momentum_pressure_forcing:
                (pressure_tendency,) = pressure_tendencies
                hydrostatic_gradient = (pressure_tendency.u, pressure_tendency.v)
            gu_cor, gv_cor, cd_state = cd_scheme.advance(
                history=self._cd_state, u=bounded.u.data, v=bounded.v.data, eta=bounded.eta.data,
                g_u=cd_g_u, g_v=cd_g_v, hydrostatic_gradient=hydrostatic_gradient, grid=grid,
                gravity=self.gravity, dt_mom=dt_mom, ab_eps=self.ab_eps,
            )
            object.__setattr__(self, "_cd_state", cd_state)
            tend_u = tend_u + gu_cor
            tend_v = tend_v + gv_cor
        # 1c. NonlinFreeSurf > 1: rescale the momentum tendency at surface cells.
        #     MITgcm timestep.F:303-317: gUtmp *= _hFacW / hFac_surfW, i.e.
        #     h(eta^{n-1}) / h(eta^n) -- the geometry still holds the lagged
        #     thickness at this point (see the time-layer note at the top of the
        #     step).  Applied to the assembled slow tendency, which in
        #     synchronous stepping already carries the AB2-weighted hydrostatic
        #     pressure gradient (timestep.F:116-125); the surface-pressure term
        #     is added by the predictor below and is NOT rescaled
        #     (timestep.F:330-333).  The AB2 memory is deliberately left alone --
        #     MITgcm rescales the tracer memory but not guNm1/gvNm1.  The
        #     vector-invariant form takes no rescale (timestep.F:275-276,
        #     ``.NOT. vectorInvariantMomentum``): its tendency is already per
        #     unit thickness.
        nlfs_momentum = self.nonlin_free_surf >= 2 and self._surface_dr_active(grid)
        if nlfs_momentum and self._resolved_vector_invariant_momentum is None:
            raise RuntimeError(
                "nonlin_free_surf>=2: the momentum form (vectorInvariantMomentum) was "
                "not resolved at model assembly (resolve_vector_invariant_momentum)"
            )
        if nlfs_momentum and not self._resolved_vector_invariant_momentum:
            _, _hfs_w, _hfs_s = self._surface_geometry.surface
            tend_u, tend_v = _rescale_tendency_nonlin_freesurf(
                tend_u, tend_v,
                hFacW_now=ifs_geometry.surface_hfac(grid.masks.hFacW),
                hFacS_now=ifs_geometry.surface_hfac(grid.masks.hFacS),
                hFac_surfW=_hfs_w,
                hFac_surfS=_hfs_s,
            )

        for pressure_tendency in pressure_tendencies:
            tend_u, tend_v = tend_u + pressure_tendency.u, tend_v + pressure_tendency.v

        # MITgcm recip_hFacNew for the implicit tracer stage; None keeps the
        # solves on the grid's own reciprocal (linear free surface).
        recip_hfac_new = None
        if not self.stagger_tracer_stepping:
            if self._r_star_active(grid):
                # The grid holds h0Fac*rStarFac(eta^{n-1}) and rStarExpC is
                # rStarFac(eta^n)/rStarFac(eta^{n-1}) (forward_step.F:732 runs
                # THERMODYNAMICS before UPDATE_R_STAR); temp_integrate.F
                # rescales the tendency and the AB memory alike.
                tend_temp, tend_salt, recip_hfac_new = ifs_geometry.rescale_r_star_tracers(
                    tend_temp, tend_salt, hfac_c=grid.masks.hFacC, exp_c=self._r_star.factors.exp_c,
                )
                history = self._momentum_ab
                if history is not None:
                    object.__setattr__(self, "_momentum_ab", ifs_tendency.rescale_tracer_ab_r_star(
                        history, exp_c=self._r_star.factors.exp_c,
                    ))
            elif (
                self.nonlin_free_surf >= 1
                and self.select_r_star == 0
                and ifs_geometry.surface_geometry_has_masks(grid.masks)
            ):
                tend_temp, tend_salt, recip_hfac_new = ifs_geometry.rescale_surface_tracers(
                    tend_temp, tend_salt, hfac_c=grid.masks.hFacC,
                    surface_hfac_c=self._surface_geometry.surface[0],
                )
                history = self._momentum_ab
                if history is not None:
                    object.__setattr__(self, "_momentum_ab", ifs_tendency.rescale_tracer_ab_surface(
                        history, hfac_c_now=ifs_geometry.surface_hfac(grid.masks.hFacC),
                        surface_hfac_c=self._surface_geometry.surface[0],
                    ))
        return tend_u, tend_v, tend_temp, tend_salt, recip_hfac_new

    def _advance_synchronous_tracers(
        self, *, bounded, grid, dt_tracer, tend_temp, tend_salt,
        recip_hfac_new, kpp, kpp_fields, mixing, gm_contributions, kpp_post_names, temperature_advection_scheme, salinity_advection_scheme, multi_dim_advection,
    ):
        tracer_temp_n = bounded.temp.data
        temp_new = ifs_tracer.advance_explicit(
            tracer=tracer_temp_n, dt_tracer=dt_tracer, tendency=tend_temp,
        )
        tracer_salt_n = bounded.salt.data
        salt_new = ifs_tracer.advance_explicit(
            tracer=tracer_salt_n, dt_tracer=dt_tracer, tendency=tend_salt,
        )
        if self.stream_trace.hook is not None:
            # The stage stations show an unstepped tracer at its step-start value:
            # MITgcm never integrates it (thermodynamics.F:317-337).
            held_temp, held_salt = self._hold_unstepped_tracers(
                bounded=bounded, temp_new=temp_new, salt_new=salt_new,
            )
            self.stream_trace.emit_stream_trace(
                "tracer.after_explicit",
                temp=held_temp,
                salt=held_salt,
            )
        _kpp_t_enabled = kpp_fields is not None
        recip_hfac_current = recip_hfac_new
        tracer_mask_current = None
        tracer_recip_current = None
        if _kpp_t_enabled:
            _kpp_layer_thickness = kpp.inputs.static.layer_thickness
            _kpp_convective_implicit = self.kpp_convective_implicit
            if recip_hfac_new is not None and _kpp_convective_implicit is not None:
                raise ValueError(
                    "recip_hfac (recip_hFacNew of a nonlinear free surface) reaches the diffusion "
                    "solve only; the implicit vertical advection and convective solves do not take it")
            if _kpp_convective_implicit is None and recip_hfac_current is None:
                tracer_mask_current, tracer_recip_current = tracer_hfac_factors(
                    field=temp_new, hfac_c=cast_tensor(grid.masks.hFacC, dtype=temp_new.dtype, device=temp_new.device),
                )
                recip_hfac_current = tracer_recip_current
            tracer_column_diffusion = False
            tracer_kappa_rt, tracer_kappa_rs = columns.assemble_tracer_diffusivity(
                tracer_kr=mixing.tracer_kr, contributions=mixing.contributions + gm_contributions)
        else:
            _kpp_layer_thickness = None
            _kpp_convective_implicit = None
            tracer_column_diffusion = self.implicit_tracer_diffusion
            tracer_kappa_rt, tracer_kappa_rs = 0.0, 0.0
            if tracer_column_diffusion:
                tracer_kappa_rt, tracer_kappa_rs = columns.assemble_tracer_diffusivity(
                    tracer_kr=mixing.tracer_kr, contributions=mixing.contributions + gm_contributions)
                if recip_hfac_current is None and grid.masks.hFacC is not None:
                    tracer_mask_current, tracer_recip_current = tracer_hfac_factors(
                        field=temp_new, hfac_c=grid.masks.hFacC,
                    )
                    recip_hfac_current = tracer_recip_current
        nonlocal_kz = (self._kpp_nonlocal_diffusivities(kpp_fields, gm_contributions)
                       if _kpp_t_enabled else (None, None))
        temp_new, salt_new = ifs_tracer.mix_and_solve_columns(
            temp_new=temp_new, salt_new=salt_new,
            temp_reference=tracer_temp_n, salt_reference=tracer_salt_n,
            kpp_enabled=_kpp_t_enabled, kpp=kpp_fields,
            layer_thickness=_kpp_layer_thickness, grid=grid, dt_tracer=dt_tracer,
            tracer_w=None,
            temperature_implicit_advection=False,
            salinity_implicit_advection=False,
            temperature_advection_scheme=temperature_advection_scheme,
            salinity_advection_scheme=salinity_advection_scheme,
            multi_dim_advection=multi_dim_advection,
            nonlocal_tracer_names=kpp_post_names,
            continuity_names=(), convective_implicit=_kpp_convective_implicit,
            implicit_tracer_diffusion=tracer_column_diffusion,
            kappa_rt=tracer_kappa_rt, kappa_rs=tracer_kappa_rs,
            nonlocal_temp_diffusivity=nonlocal_kz[0], nonlocal_salt_diffusivity=nonlocal_kz[1],
            recip_hfac=recip_hfac_current,
            mask_c_current=tracer_mask_current,
            recip_hfac_current_c=tracer_recip_current,
            recip_hfac_continuity=tracer_recip_current,
        )
        if self.stream_trace.hook is not None:
            held_temp, held_salt = self._hold_unstepped_tracers(
                bounded=bounded, temp_new=temp_new, salt_new=salt_new,
            )
            self.stream_trace.emit_stream_trace(
                "tracer.after_implicit",
                temp=held_temp,
                salt=held_salt,
            )
        return temp_new, salt_new

    def _hold_unstepped_tracers(self, *, bounded, temp_new, salt_new):
        # tempStepping / saltStepping gate all of TEMP_INTEGRATE / SALT_INTEGRATE
        # (thermodynamics.F:317-337), so an unstepped tracer leaves the explicit
        # update and the column solves with its value at the start of the step.
        if not self.temp_stepping:
            temp_new = bounded.temp.data
        if not self.salt_stepping:
            salt_new = bounded.salt.data
        return temp_new, salt_new

    def _mask_predictors(self, *, grid, u_star, v_star, temp_new, salt_new):
        # Apply masks to predicted fields
        u_mask = None
        v_mask = None
        c_mask = None
        if grid.masks.has_3d_masks:
            u_mask = grid.masks.u_float(dtype=u_star.dtype, device=u_star.device)
            v_mask = grid.masks.v_float(dtype=v_star.dtype, device=v_star.device)
            c_mask = grid.masks.center_float(dtype=temp_new.dtype, device=temp_new.device)
            if not self.momentum_implicit_vertical_advection:
                u_star, v_star = mask_velocity_pair(u_star, v_star, u_mask=u_mask, v_mask=v_mask)
            temp_new, salt_new = ifs_tracer.mask_tracer_pair(
                temp_new, salt_new, mask_c=c_mask,
            )
        return u_star, v_star, temp_new, salt_new, u_mask, v_mask, c_mask

    def _apply_predictor_boundaries(
        self, *, state, grid, bounded, boundary, enforce_velocity, enforce_boundary_tracers,
        u_star, v_star, temp_new, salt_new,
    ):
        # 3. Apply boundary conditions to predicted state.
        #    MITgcm: OBCS_APPLY_UV(gU,gV) in dynamics.F L610 uses future-time
        #    OBC values (precomputed by OBCS_CALC(myTime+dt) in do_oceanic_phys).
        #    Switch to futureTime for this and all subsequent enforce_boundaries.
        #    The sample comes off the plan (``StepPlan.boundary``), so the step
        #    body installs tensors it was handed rather than fetching them from
        #    the model.
        if boundary.future.sampled:
            boundary.future.install()
        else:
            raise RuntimeError("regional step requires declared application samples")
        predicted = state.with_tensors(
            eta=bounded.eta.data,
            u=u_star, v=v_star,
            temp=temp_new, salt=salt_new,
        )
        apply_predicted = enforce_velocity if self.stagger_tracer_stepping else enforce_boundary_tracers
        predicted_bounded, u_star, v_star = ifs_publication.apply_boundary_and_sync(
            predicted, grid, apply_boundary=apply_predicted,
            boundary_begin="predicted enforce_boundaries begin",
            boundary_done="predicted enforce_boundaries done",
            sync_begin="predicted shared velocity sync begin",
            sync_done="predicted shared velocity sync done",
        )
        temp_new = predicted_bounded.temp.data
        salt_new = predicted_bounded.salt.data
        if ckpt_dump.ENABLED:
            record_checkpoint(
                "obcs", u=u_star, v=v_star, temp=temp_new, salt=salt_new
            )
        return u_star, v_star, temp_new, salt_new

    def _prepare_vertical_momentum_state(self, *, bounded, grid, u_star):
        cached_w_vel = self._nh_history.w_vel
        w_vel_mit = ifs_nh.prepare_nh_w_velocity(
            bounded, previous_w_vel=cached_w_vel, u_star=u_star, grid=grid,
        )
        del cached_w_vel
        phi_nh_history = self._nh_history.phi_nh
        return w_vel_mit, phi_nh_history

    def _prepare_vertical_momentum_history(self, *, phi_nh_history, temp_new):
        phi_nh_prev = cast_tensor(phi_nh_history,
            dtype=temp_new.dtype, device=temp_new.device,
        )
        gw_prev = self._nh_history.gw_nm1
        if gw_prev is not None:
            gw_prev = cast_tensor(gw_prev, dtype=temp_new.dtype, device=temp_new.device)
        return phi_nh_prev, gw_prev

    def _prepare_surface_transport(
        self, *, grid, u_star, v_star, bounded, eta_h, hydro_history, empmr_mass_flux,
    ):
        # MITgcm UPDATE_SURF_DR(.TRUE.) at forward_step.F:852 (MITgcm 27256b2d) -- the geometry
        # catches up with the free surface *here*, after THERMODYNAMICS and
        # DYNAMICS and before the divergence/Helmholtz/continuity block, so
        # everything below runs on h(eta^n).  With nonlinFreeSurf > 2 the CG2D
        # coefficients are rebuilt from it every step (MITgcm UPDATE_CG2D,
        # forward_step.F:869 at 27256b2d; _get_coefficients skips its cache).
        if self._surface_dr_active(grid):
            self._apply_surf_dr(grid)
        elif self._r_star_active(grid):
            # UPDATE_R_STAR(.TRUE.) at forward_step.F:832-840.
            ifs_geometry.install_r_star_hfac(grid, self._r_star.columns, self._r_star.factors)

        (
            div_transport_star, layer_thickness, hFacW, hFacS,
        ) = ifs_freshwater.regional_predictor_transport(
            u_star, v_star, grid, stream_trace=self.stream_trace)

        etaH_prev = (hydro_history.eta_h if hydro_history is not None
                     else (bounded.eta.data if eta_h is None else eta_h))
        surface_mask, maskInC_3d, empmr_mass_flux = ifs_freshwater.prepare_surface_rhs(
            etaH_prev, empmr_mass_flux, grid=grid,
        )
        return (div_transport_star, layer_thickness, hFacW, hFacS, etaH_prev,
                surface_mask, maskInC_3d, empmr_mass_flux)

    def _prepare_surface_elevation_source(
        self, *, grid, etaH_prev, old_free_surf_term, phi_nh_prev,
    ):
        # MITgcm ``oldFreeSurfTerm = use3Dsolver .AND. .NOT.exactConserv``
        # (solve_for_pressure.F:86); the non-hydrostatic half it then adds to
        # the cg2d source term is assembled in ``ifs_freshwater.nh_free_surface_source``.
        # Two things are decided here because only the stepper knows them:
        #
        #  * Time level -- MITgcm reads ``phi_nh`` as it stands on entry to
        #    SOLVE_FOR_PRESSURE, i.e. the previous step's CG3D solution.  That
        #    is exactly ``phi_nh_prev`` above (this step's CG3D runs after the
        #    cg2d solve below and only then rewrites the cache).
        #  * ``recip_Bo`` -- ``Bo_surf = gBaro`` for z-coordinates
        #    (ini_linear_phisurf.F:84) and LOCOMgcm carries no separate gBaro, so it
        #    is ``1/gravity``, the same value ``_get_cg3d_coefficients`` uses
        #    for the matching CG3D surface diagonal.
        #
        # Gated on the *internal* NH solve: the external phi_nh provider never
        # populates ``_nh_history.phi_nh`` (it is called once, with the end-of-step
        # time, and feeds only the U/V correction), so no previous-step phi_nh
        # exists to read on that path.
        # MITgcm ``solve_for_pressure.F:85``.  One definition feeds both the
        # cg2d source term here and the cg3d side below.
        nh_surface_source = None
        eta_source = etaH_prev
        if old_free_surf_term:
            nh_surface_source, eta_source = ifs_freshwater.nh_free_surface_source(
                phi_nh_prev, eta_h=etaH_prev, gravity=self.gravity,
                masks=grid.masks, ny=grid.ny, nx=grid.nx,
            )
        return nh_surface_source, eta_source

    def _retain_surface_elevation(self, *, eta_new):
        if ckpt_dump.ENABLED:
            record_checkpoint("cg_rhs")
            record_checkpoint("cg_sol", eta=eta_new)

        # MITgcm STDOUT monitors CG-solved etaN rather than post-exactConserv etaH.
        eta_n = eta_new.detach().clone()
        # Multi-step diagnostic printf (env-gated, scalar-only).
        # Prints eta_new global max + location each step.  Pure `.item()` calls,
        # no GPU pressure.  Activate via R4_TRACK=1.
        if _R4_TRACK_ENABLED:
            track_surface_solution(eta_new, step_index=self.stream_trace.step_index)
        return eta_n

    def _correct_barotropic_velocity(
        self, *, eta_new, u_star, v_star, grid, alpha, dt_mom, surface_mask,
    ):
        # 7. Correct velocities with new surface pressure gradient
        # u^{n+1} = u* - α·g·dt·∂η^{n+1}/∂x
        # MITgcm: psFac = pfFacMom * implicSurfPress; uVel = gU - psFac*dt*phiSurfX
        # Unified with the cube step body in ImplicitFreeSurfaceStepper._correct_surface_velocity.
        # The two regional-only mask *constructions* stay here; the
        # segment applies them to the η gradient (so a masked face keeps u*, unlike
        # the cube's result masking which zeroes it — hence u_mask/v_mask=None).

        # Mask gradients at wet-dry boundary faces to be consistent with
        # the Helmholtz operator (which zeros off-diagonal coefficients there).
        # Without this, the correction creates spurious velocity at faces
        # between wet and dry cells, producing transport inconsistent with
        # the solved η. (MITgcm: _maskW applied to phiSurfX in correction_step.F)
        uf = None
        vf = None
        if grid.masks.has_3d_masks:
            uf, vf = surface_pressure_face_masks(
                surface_mask[0], ny=grid.ny, nx=grid.nx,
            )

        # Mask velocity correction at OB-adjacent faces (maskInW)
        # MITgcm: correction_step.F multiplies phiSurfX by maskInW,
        # preventing the implicit PG correction at OB faces. Without this,
        # the large ∂η/∂x at the OB face (where η=0 is forced) generates
        # a large spurious correction that is then overwritten by
        # enforce_boundaries, creating inconsistent transport in exactConserv.
        maskInW = None
        maskInS = None
        if grid.boundary_topology.open_x or grid.boundary_topology.open_y:
            # dtype/device were read off the η gradient before it moved into the
            # segment; eta_new is that gradient's allocation source.
            _, maskInW, maskInS = _compute_interior_mask(
                grid, eta_new.dtype, eta_new.device,
            )

        u_new, v_new = self._correct_surface_velocity(
            eta_new,
            u_star,
            v_star,
            grid,
            alpha=alpha,
            dt_mom=dt_mom,
            dtype=eta_new.dtype,
            device=eta_new.device,
            wet_face_mask_u=uf,
            wet_face_mask_v=vf,
            mask_in_w=maskInW,
            mask_in_s=maskInS,
        )
        return u_new, v_new

    def _prepare_internal_pressure(
        self, *, u_star, v_star, grid, model, bundle,
        dt_mom, dt_freesurf, temp_new, phi_nh_history, eta_new, etaH_prev,
        w_vel_new_mit_for_cg3d, old_free_surf_term, nh_surface_source,
    ):
        _dist_trace("internal nonhydrostatic solve begin")
        if w_vel_new_mit_for_cg3d is None:
            raise RuntimeError("internal NH path missing MIT-convention transient wVel")
        (
            cg3d_b_init, phi_nh_init, mass_to_r_unit_value,
            cg2d_x_solution, maskInC_for_nh,
        ) = ifs_nh.prepare_internal_pressure(
            u_star=u_star, v_star=v_star, grid=grid, dt_mom=dt_mom,
            implicit_div_2d_flow=self.implicit_div_2d_flow, template=temp_new,
            phi_nh_history=phi_nh_history,
            mass_to_r_unit=self._prepared_mass_to_r_unit,
            gravity=self.gravity, eta_new=eta_new,
        )
        if old_free_surf_term:
            assert nh_surface_source is not None
            cg3d_b_init = ifs_nh.add_old_free_surface_source(
                cg3d_b_init=cg3d_b_init, grid=grid, eta_h_prev=etaH_prev,
                nh_surface_source=nh_surface_source,
                dt_mom=dt_mom, dt_freesurf=dt_freesurf,
            )
        cg3d_b, phi_nh_init_masked, nh_mask = ifs_nh.assemble_internal_pressure_rhs(
            cg3d_b_init=cg3d_b_init, cg2d_x_solution=cg2d_x_solution,
            eta_new=eta_new, etaH_prev=etaH_prev,
            w_vel_new_mit_for_cg3d=w_vel_new_mit_for_cg3d,
            phi_nh_init=phi_nh_init, grid=grid, bundle=bundle,
            params=self.nonhydrostatic, old_free_surf_term=old_free_surf_term,
            mass_to_r_unit_value=mass_to_r_unit_value,
            implicit_surf_press=self.implicit_surf_press,
            implicit_div_2d_flow=self.implicit_div_2d_flow,
            template=temp_new, maskInC_for_nh=maskInC_for_nh,
        )
        # Fixed geometry: built in prepare (``_prepare_cg3d_coefficients``).
        prepared_cg3d = getattr(self, "_prepared_cg3d_coefficients", None)
        operator, preconditioner = (
            prepared_cg3d
            if prepared_cg3d is not None
            else self._get_cg3d_coefficients(
                grid,
                bundle,
                temp_new.dtype,
                temp_new.device,
                mask_in_c=maskInC_for_nh,
            )
        )
        return cg3d_b, phi_nh_init_masked, nh_mask, operator, preconditioner

    def _complete_internal_pressure(
        self, *, u_new, v_new, grid, dt_mom, phi_nh_new, surface_mask,
    ):
        # correction_step.F:157: nhFac = pfFacMom * implicitNHPress.
        # The supported internal NH path fixes implicitNHPress at one.
        if self.momentum_pressure_forcing:
            u_new, v_new = pressure.apply_pnh_correction_from_phi(
                u_new=u_new,
                v_new=v_new,
                grid=grid,
                dt_mom=dt_mom,
                phi=phi_nh_new,
                surface_mask=surface_mask if grid.masks.has_3d_masks else None,
            )
        _dist_trace("internal nonhydrostatic solve done")
        return u_new, v_new

    def _apply_corrected_boundaries(
        self, *, state, grid, u_new, v_new, eta_new, temp_new, salt_new,
        surface_mask, dt_mom, u_mask, v_mask, boundary, enforce_velocity,
    ):
        # External nonHydrostatic phi_nh correction after eta-pressure
        # correction and before masks, enforce_boundaries, and integr_continuity.
        # Provider wins whenever present, regardless of the internal NH flag.
        u_new, v_new = self._apply_pnh_correction(
            u_new,
            v_new,
            grid,
            surface_mask if grid.masks.has_3d_masks else None,
            dt_mom,
        )

        if ckpt_dump.ENABLED:
            record_checkpoint("correction", u=u_new, v=v_new)

        # Apply masks
        if grid.masks.has_3d_masks:
            u_new, v_new = mask_velocity_pair(u_new, v_new, u_mask=u_mask, v_mask=v_mask)

        # 8. Apply boundary conditions to corrected velocities BEFORE
        #    computing exactConserv. MITgcm step order:
        #    CORRECTION_STEP → OBCS_APPLY_UV(uVel,vVel) → INTEGR_CONTINUITY
        #    The exactConserv transport divergence must use post-boundary
        #    velocities so that prescribed OB transport is included.
        corrected = state.with_tensors(
            eta=eta_new,  # CG solution η (MITgcm: etaN after solve_for_pressure)
            u=u_new,
            v=v_new,
            temp=temp_new,
            salt=salt_new,
        )
        if boundary.future.sampled:
            boundary.future.install()
        else:
            raise RuntimeError("regional step requires declared application samples")
        corrected_bounded, u_final, v_final = ifs_publication.apply_boundary_and_sync(
            corrected, grid, apply_boundary=enforce_velocity,
            boundary_begin="corrected enforce_boundaries begin",
            boundary_done="corrected enforce_boundaries done",
            sync_begin="corrected shared velocity sync begin",
            sync_done="corrected shared velocity sync done",
        )
        return corrected_bounded, u_final, v_final

    def _advance_continuity(
        self, *, state, grid, bounded, corrected_bounded, u_final, v_final,
        eta_new, etaH_prev, beta, dt_freesurf, layer_thickness, hFacW, hFacS,
        empmr_mass_flux, mass_to_r_unit, boundary, hydro_history,
        implicit_vertical_advection_active,
    ):
        self.stream_trace.emit_stream_trace(
            "step.post_momentum_correction",
            u=u_final,
            v=v_final,
            eta=eta_new,
        )

        if self.exact_conserv:
            _dist_trace("exactConserv begin")
            d_eta_h_dt = ifs_freshwater.regional_continuity_rate(
                u_final=u_final, v_final=v_final, layer_thickness=layer_thickness,
                hFacW=hFacW, hFacS=hFacS, grid=grid,
            )
            d_eta_dt_times_dt = dt_freesurf * d_eta_h_dt
            if self._r_star_active(grid):
                # integr_continuity.F:245-256: the rate of this call's w, which
                # every w diagnosed from u_final, v_final reads.
                grid.masks.set_r_star_thickness_rate(r_star.r_star_thickness_rate(
                    d_eta_h_dt, self._r_star.columns.recip_depth_c))
            if self.use_real_freshwater_flux and empmr_mass_flux is not None:
                d_eta_dt_times_dt = ifs_freshwater.subtract_continuity_freshwater(
                    d_eta_dt_times_dt, mass_to_r_unit=mass_to_r_unit,
                    empmr_mass_flux=empmr_mass_flux, dt_freesurf=dt_freesurf,
                )
            eta_final, eta_H_new = ifs_freshwater.advance_continuity_surface(
                etaH_prev, beta=beta, d_eta_dt_times_dt=d_eta_dt_times_dt,
            )
            if self._uses_freshwater_history():
                pmep_r_new = ifs_freshwater.continuity_freshwater_history(
                    empmr_mass_flux, grid=grid,
                )
                object.__setattr__(self, "_pmep_r", pmep_r_new)
            # UPDATE_ETAH preserves land and OB cells instead of masking them
            # to zero. Nonlinear runs then overwrite wet OB etaH cells with
            # the application ETA sample below (update_etah.F:74-75).

            # No domain-mean eta projection here: MITgcm has none, and under
            # useOBCSbalance the boundary condition's normal-velocity correction
            # (obcs_balance_flow.F) keeps the net open-boundary inflow at zero.
            _dist_trace("exactConserv done")
        else:
            eta_final = eta_new
            eta_H_new = eta_new

        if self._surface_dr_active(grid):
            # Every UPDATE_ETAH applies OBCS_APPLY_ETA before etaH is saved or
            # CALC_SURF_DR reads it, so center and face thicknesses share ETA.
            eta_H_new = apply_surface_eta_boundary(
                eta_H_new, grid, obc_eta=boundary.future.obc_eta,
            )

        if ckpt_dump.ENABLED:
            from defgcm.dynamics.vertical_velocity import diagnose_vertical_velocity

            _continuity_state = corrected_bounded.with_tensors(
                eta=eta_final,
                u=u_final,
                v=v_final,
            )
            _, _continuity_w, _ = diagnose_vertical_velocity(_continuity_state, grid)
            record_checkpoint("continuity", eta=eta_final, w=_continuity_w)

        eta_h_history = None
        if hydro_history is None:
            eta_h_history = _carry(eta_H_new).clone()

        # MITgcm CALC_SURF_DR(etaH) at forward_step.F:959 (MITgcm 27256b2d) -- hFac_surf* moves to
        # h(eta^{n+1}) while the geometry stays at h(eta^n), which is exactly the
        # lag the next step's rescale factors need.
        if self._surface_dr_active(grid):
            object.__setattr__(self, "_surface_geometry", self._refresh_hfac_surf(
                grid, eta_H_new, obc_eta=boundary.future.obc_eta,
            ))
        elif self._r_star_active(grid):
            # CALC_R_STAR(etaH) at forward_step.F:939-950: factors of
            # eta^{n+1}, while the grid keeps h0Fac*rStarFac(eta^n).
            self._calc_r_star(grid, eta_H_new, dt_freesurf=dt_freesurf)

        if self.stream_trace.hook is not None:
            from defgcm.dynamics.vertical_velocity import diagnose_vertical_velocity

            continuity_state = corrected_bounded.with_tensors(
                eta=eta_final,
                u=u_final,
                v=v_final,
            )
            _, continuity_w, _ = diagnose_vertical_velocity(continuity_state, grid)
            self.stream_trace.emit_stream_trace(
                "step.post_integr_continuity",
                w=continuity_w,
                eta=eta_final,
            )

        tracer_w = None
        if implicit_vertical_advection_active:
            from defgcm.dynamics.implicit_vertical_advection import (
                make_tracer_w_np_half_mit,
            )

            corrected_velocity_state = state.with_tensors(
                eta=eta_final,
                u=u_final,
                v=v_final,
                temp=bounded.temp.data,
                salt=bounded.salt.data,
            )
            tracer_w = make_tracer_w_np_half_mit(corrected_velocity_state, grid)
        return eta_final, eta_H_new, eta_h_history, tracer_w

    def _select_tracer_terms(self):
        # ── Staggered tracer stepping ──────────────────────────────────
        # MITgcm staggerTimeStep: THERMODYNAMICS runs AFTER velocity
        # correction, so tracers see u^{n+1}. Build a staggered state
        # with corrected velocity + old tracers, evaluate tracer modules,
        # then advance tracers.
        implicit_tracer_names = tuple(
            name
            for name, enabled in (
                ("temp", self.temperature_implicit_vertical_advection),
                ("salt", self.salinity_implicit_vertical_advection),
            )
            if enabled
        )
        continuity_ab_names = tuple(
            name for name, enabled in (
                ("temp", self.ab2_on_temperature_tendency),
                ("salt", self.ab2_on_salinity_tendency),
            ) if name in implicit_tracer_names and self.ab_eps > 0.0 and enabled
        )
        continuity_post_names = tuple(
            name for name in implicit_tracer_names if name not in continuity_ab_names
        )
        return implicit_tracer_names, continuity_ab_names, continuity_post_names

    def _evaluate_staggered_tracers(
        self, *, state, model, grid, bounded, eta_final, u_final, v_final,
        dt_tracer, evaluation_inputs, kpp, kpp_fields, kpp_ab_names, kpp_nonlocal,
        implicit_tracer_names, continuity_ab_names, tracer_w,
    ):
        _dist_trace("staggered tracer update begin")
        staggered_state = state.with_tensors(
            eta=eta_final,
            u=u_final,
            v=v_final,
            temp=bounded.temp.data,
            salt=bounded.salt.data,
        )
        # SurfaceHeatSaltFlux diagnoses CALC_WSURF_TR from this state's
        # corrected U/V. The synchronous evaluation uses its old U/V;
        # neither branch needs an eta-difference seed.

        # MIT temp_integrate.F: AB2 over advection+diffusion only; surface
        # flux + sponge are added AFTER AB2 with raw current-time values
        # (tracForcingOutAB=1).  Either per-tracer AB gate selects the split
        # route; each tracer is extrapolated and cached independently below.
        split_available = (
            (
                self.ab2_on_temperature_tendency
                or self.ab2_on_salinity_tendency
            )
            and (
                implicit_tracer_names
                or hasattr(model, "evaluate_tracer_tendency_split_ab2")
            )
        )

        if split_available:
            if implicit_tracer_names:
                tracer_sources = self._tracer_tendency_sources(model)
                tracer_ab_modules, tracer_post_modules = AssemblyPolicy.split_tracers(
                    tracer_sources.modules)
                ab2_tendency, post_ab2_tendency, tracer_diag = (
                    ifs_tendency._evaluate_tracer_tendency_for_implicit_vertical_advection(
                        staggered_state,
                        sources=tracer_sources,
                        implicit_tracer_names=implicit_tracer_names,
                        ab2_modules=tracer_ab_modules,
                        post_modules=tracer_post_modules,
                        inputs=evaluation_inputs,
                    )
                )
                del tracer_sources, tracer_ab_modules, tracer_post_modules
            else:
                ab2_tendency, post_ab2_tendency, tracer_diag = (
                    model.evaluate_tracer_tendency_split_ab2(
                        staggered_state,
                        inputs=evaluation_inputs,
                    )
                )
            model.accumulate_tendency_diagnostics(tracer_diag, dt_tracer)
            tracer_w = self._gm_residual_tracer_w(tracer_diag, tracer_w)

            if kpp_ab_names:
                temp_kz, salt_kz = self._kpp_nonlocal_diffusivities(
                    kpp_fields,
                    self._gm_kwz_contributions(
                        tracer_diag, grid, dtype=kpp_fields.diff_kzt.dtype,
                        device=kpp_fields.diff_kzt.device,
                    ) if self.kpp_ghat_use_total_diffus else (),
                )
                ab2_tendency, kpp_nonlocal = ifs_tendency._add_kpp_nonlocal_to_tracer_bucket(
                    ab2_tendency, state=staggered_state, kpp=kpp_fields,
                    temp_diffusivity=temp_kz, salt_diffusivity=salt_kz,
                    layer_thickness=kpp.inputs.static.layer_thickness,
                    grid=grid, tracer_names=kpp_ab_names,
                )

            if continuity_ab_names:
                from defgcm.dynamics.implicit_vertical_advection import (
                    TracerVerticalContinuity,
                )

                continuity_module = TracerVerticalContinuity(tracer_w, continuity_ab_names)
                if AssemblyPolicy.tracer_bucket_of(continuity_module) != AssemblyBucket.AB2:
                    raise ValueError("Tracer continuity must belong to the GAD AB bucket")
                continuity = continuity_module.evaluate(staggered_state, DiagnosticState(), grid)
                # gad_calc_rhs.F includes C before AB and its history write.
                # Euler fields keep the old state + dt*C order downstream.
                ab2_tendency = StateTendency(
                    eta=ab2_tendency.eta, u=ab2_tendency.u, v=ab2_tendency.v,
                    temp=(ifs_tendency.add_tendency_term(ab2_tendency.temp, continuity.temp)
                          if "temp" in continuity_ab_names else ab2_tendency.temp),
                    salt=(ifs_tendency.add_tendency_term(ab2_tendency.salt, continuity.salt)
                          if "salt" in continuity_ab_names else ab2_tendency.salt),
                )

            # AB2 only on the ab2_eligible bucket (adv + diff).
            if (self.ab_eps > 0.0 and self._tracer_ab is not None
                    and not self._tracer_ab.reserved):
                prev_tr = self._tracer_ab
                ab_fac = ifs_tendency.ab2_factor(self.ab_eps)
                if self.ab2_on_temperature_tendency and prev_tr.temp is not None:
                    tt_ab = ifs_tendency.extrapolate_ab2(
                        ab2_tendency.temp, prev_tr.temp, factor=ab_fac,
                    )
                else:
                    tt_ab = ab2_tendency.temp
                if self.ab2_on_salinity_tendency and prev_tr.salt is not None:
                    ts_ab = ifs_tendency.extrapolate_ab2(
                        ab2_tendency.salt, prev_tr.salt, factor=ab_fac,
                    )
                else:
                    ts_ab = ab2_tendency.salt
            else:
                tt_ab = ab2_tendency.temp
                ts_ab = ab2_tendency.salt
            if self.ab_eps > 0.0:
                tracer_cache = {}
                if self.ab2_on_temperature_tendency:
                    tracer_cache["temp"] = ifs_tracer.retain_tracer_ab(
                        ab2_tendency.temp,
                    )
                if self.ab2_on_salinity_tendency:
                    tracer_cache["salt"] = ifs_tracer.retain_tracer_ab(
                        ab2_tendency.salt,
                    )
                object.__setattr__(self, "_tracer_ab", ifs_tracer.TracerABHistory(
                    tracer_cache.get("temp"), tracer_cache.get("salt"), False,
                ))
            # MIT temp/salt_integrate retain GAD AB before GAD_IMPLICIT_R;
            # implicit advection only removes the explicit vertical flux.
            # Add raw surface flux and sponge tendencies after AB.
            if post_ab2_tendency is not None:
                tracer_temp_tendency = ifs_tendency.add_tendency_term(tt_ab, post_ab2_tendency.temp)
                tracer_salt_tendency = ifs_tendency.add_tendency_term(ts_ab, post_ab2_tendency.salt)
            else:
                tracer_temp_tendency = tt_ab
                tracer_salt_tendency = ts_ab
        else:
            # Debug/fallback path: single lump Euler (or legacy whole-lump AB2).
            if implicit_tracer_names:
                tracer_sources = self._tracer_tendency_sources(model)
                tracer_tendency, _, tracer_diag = (
                    ifs_tendency._evaluate_tracer_tendency_for_implicit_vertical_advection(
                        staggered_state,
                        sources=tracer_sources,
                        implicit_tracer_names=implicit_tracer_names,
                        ab2_modules=tracer_sources.modules,
                        post_modules=(),
                        inputs=evaluation_inputs,
                    )
                )
                del tracer_sources
            else:
                tracer_tendency, tracer_diag = model.evaluate_tracer_tendency_with_diagnostics(
                    staggered_state,
                    inputs=evaluation_inputs,
                )
            model.accumulate_tendency_diagnostics(tracer_diag, dt_tracer)
            tracer_w = self._gm_residual_tracer_w(tracer_diag, tracer_w)
            tracer_temp_tendency = tracer_tendency.temp
            tracer_salt_tendency = tracer_tendency.salt
        recip_hfac_surf_future = None
        tracer_nlfs = self._nonlin_free_surf_active(grid)
        if self._r_star_active(grid):
            # forward_step.F:832-840 installs h0Fac*rStarFac(eta^n); :939-950
            # computes rStarExpC = rStarFac(eta^{n+1})/rStarFac(eta^n) before
            # the staggered THERMODYNAMICS; FREESURF_RESCALE_G divides the
            # tendency and the freshly stored AB memory at every level.
            tracer_temp_tendency, tracer_salt_tendency, recip_hfac_surf_future = (
                ifs_geometry.rescale_r_star_tracers(
                    tracer_temp_tendency, tracer_salt_tendency,
                    hfac_c=grid.masks.hFacC, exp_c=self._r_star.factors.exp_c,
                )
            )
            if split_available and self.ab_eps > 0.0:
                object.__setattr__(self, "_tracer_ab", ifs_tendency.rescale_tracer_ab_r_star(
                    self._tracer_ab, exp_c=self._r_star.factors.exp_c,
                ))
        elif tracer_nlfs:
            # forward_step.F:792 runs DYNAMICS; :831-853 installs h(eta^n).
            # :958-960 computes hFac_surf(eta^(n+1)) before stagger
            # THERMODYNAMICS :976-1006. temp_integrate.F:411,433 rescales
            # the full tendency and freshly stored raw AB history before
            # TIMESTEP_TRACER :469-473.
            hfc_now = ifs_geometry.surface_hfac(grid.masks.hFacC)
            hfs_c = self._surface_geometry.surface[0]
            tracer_temp_tendency, tracer_salt_tendency, recip_hfac_surf_future = (
                ifs_geometry.rescale_surface_tracers(
                    tracer_temp_tendency, tracer_salt_tendency,
                    hfac_c=grid.masks.hFacC, surface_hfac_c=hfs_c,
                )
            )
            if kpp_ab_names and self.kpp_tracer_substeps > 1:
                # Substeps remove raw N^n from this rescaled tendency.
                # Use its rescaled volume before adding refreshed N.
                kpp_nonlocal = ifs_tendency.rescale_kpp_nonlocal(
                    kpp_nonlocal, tracer_names=kpp_ab_names,
                    hfac_c_now=hfc_now, surface_hfac_c=hfs_c,
                )
            if split_available and self.ab_eps > 0.0:
                object.__setattr__(self, "_tracer_ab", ifs_tendency.rescale_tracer_ab_surface(
                    self._tracer_ab, hfac_c_now=hfc_now, surface_hfac_c=hfs_c,
                ))
        gm_contributions = self._gm_kwz_contributions(
            tracer_diag, grid, dtype=tracer_temp_tendency.dtype, device=tracer_temp_tendency.device)
        return (tracer_temp_tendency, tracer_salt_tendency, kpp_nonlocal,
                tracer_nlfs, recip_hfac_surf_future, gm_contributions, tracer_w)

    def _prepare_staggered_tracer_prediction(
        self, *, tracer_temp_n, tracer_salt_n, dt_tracer, tracer_temp_tendency, tracer_salt_tendency,
    ):
        temp_new = ifs_tracer.advance_explicit(
            tracer=tracer_temp_n, dt_tracer=dt_tracer, tendency=tracer_temp_tendency,
        )
        salt_new = ifs_tracer.advance_explicit(
            tracer=tracer_salt_n, dt_tracer=dt_tracer, tendency=tracer_salt_tendency,
        )
        return temp_new, salt_new

    def _prepare_subcycle_tracer_inputs(
        self, *, bounded, model, grid, kpp_forcing, freezing_flux, temp_new,
        tracer_nlfs, recip_hfac_surf_future,
    ):
        # Keep the tracer station's pressure sampling after the outer
        # dynamics; it can differ from the momentum KPP pressure above.
        substep_kpp_inputs = self._prepare_regional_kpp_inputs(
            bounded, model, grid,
            bounded.temp.data.dtype, bounded.temp.data.device,
            forcing=kpp_forcing, freezing_flux=freezing_flux,
        )
        tracer_mask_current, tracer_recip_current = tracer_hfac_factors(
            field=temp_new,
            hfac_c=cast_tensor(grid.masks.hFacC, dtype=temp_new.dtype, device=temp_new.device),
        )
        recip_hfac_tracer_matrix = (
            recip_hfac_surf_future if tracer_nlfs else tracer_recip_current
        )
        return (substep_kpp_inputs, tracer_mask_current, tracer_recip_current,
                recip_hfac_tracer_matrix)

    def _prepare_tracer_column_inputs(
        self, *, grid, temp_new, salt_new, kpp, mixing, gm_contributions,
        implicit_tracer_names, tracer_w, tracer_nlfs, recip_hfac_surf_future,
    ):
        if implicit_tracer_names and tracer_w is None:
            raise RuntimeError(
                "implicit tracer vertical advection lacks tracer-time wrapper"
            )
        _kpp_ts_enabled = kpp is not None
        tracer_mask_current = None
        tracer_recip_current = None
        if _kpp_ts_enabled:
            _kpp_layer_thickness = kpp.inputs.static.layer_thickness
            _kpp_convective_implicit = self.kpp_convective_implicit
            tracer_column_diffusion = False
            tracer_kappa_rt, tracer_kappa_rs = columns.assemble_tracer_diffusivity(
                tracer_kr=mixing.tracer_kr, contributions=mixing.contributions + gm_contributions)
            if _kpp_convective_implicit is None:
                tracer_mask_current, tracer_recip_current = tracer_hfac_factors(
                    field=temp_new, hfac_c=cast_tensor(grid.masks.hFacC, dtype=temp_new.dtype, device=temp_new.device),
                )
        else:
            _kpp_layer_thickness = None
            _kpp_convective_implicit = None
            tracer_column_diffusion = self.implicit_tracer_diffusion
            tracer_kappa_rt, tracer_kappa_rs = 0.0, 0.0
            if tracer_column_diffusion:
                tracer_kappa_rt, tracer_kappa_rs = columns.assemble_tracer_diffusivity(
                    tracer_kr=mixing.tracer_kr, contributions=mixing.contributions + gm_contributions)
            if implicit_tracer_names:
                tracer_matrix_field = temp_new if "temp" in implicit_tracer_names else salt_new
                tracer_mask_current, tracer_recip_current = tracer_hfac_factors(
                    field=tracer_matrix_field, hfac_c=getattr(grid.masks, "hFacC", None),
                )
            elif tracer_column_diffusion and grid.masks.hFacC is not None:
                tracer_mask_current, tracer_recip_current = tracer_hfac_factors(
                    field=temp_new, hfac_c=grid.masks.hFacC,
                )
        # thermodynamics.F:236-244 passes recip_hFacNew to
        # GAD_IMPLICIT_R at temp_integrate.F:481,489-494.
        recip_hfac_tracer_matrix = (
            recip_hfac_surf_future if tracer_nlfs else tracer_recip_current
        )
        return (_kpp_ts_enabled, _kpp_layer_thickness, _kpp_convective_implicit,
                tracer_column_diffusion, tracer_kappa_rt, tracer_kappa_rs,
                tracer_mask_current, tracer_recip_current, recip_hfac_tracer_matrix)

    def _step_impl(
        self,
        state: PrognosticState,
        model: SplitStepperModelContext,
        plan: "StepPlan",
        *,
        ab_history: dict[str, torch.Tensor] | None = None,
        hydro_history=None,
        boundary_evaluation=None,
    ) -> StepOutcome:
        """Advance one time step with semi-implicit free surface.

        ``plan`` supplies the sampled boundary fields, time scales and prepared
        KPP inputs. Regional KPP reads its inputs at the enabled forcing station.

        Steps:
        1. Compute slow tendencies and apply AB2 extrapolation
        2. Predict u* = u^n + dt*(AB2_tend - (1-α)·g·∇η^n)
        3. Apply boundary conditions to predicted state
        4. Compute transport divergence from predictions
        5. Assemble RHS = η^n - β·dt·div(transport*)
        6. Solve Helmholtz: [-α·β·g·dt_mom·dt_freesurf·∇·(H·∇) + I]η^{n+1} = RHS
        7. Correct: u^{n+1} = u* - α·g·dt·∇η^{n+1}
        8. Recompute η from corrected velocity divergence (exactConserv)
        9. Apply boundary conditions to final state
        """
        (implicit_vertical_advection_active, bundle, dt_mom, dt_tracer,
         dt_freesurf, grid, alpha, beta, temperature_advection_scheme,
         salinity_advection_scheme, multi_dim_advection) = self._prepare_step_entry(model=model, plan=plan)

        # Cubed-sphere global path.  The regional ``step()`` below assumes a
        # single-patch ``[batch, nz, ny, nx]`` layout (1-D row metrics, ``dim=1``
        # nz reductions, ``surface_mask[0:1]`` batch indexing) that does not hold
        # on the FACE-FIRST cube state ``[6, nz, ny, nx]``.  Route the cube to a
        # dedicated, self-contained step that reuses the already cube-ready
        # helpers (``_assemble_helmholtz_coefficients`` / ``cg2d_solve``)
        # and the cube dynamics modules.
        # Guarded by ``face_count == 6`` so the regional path is byte-identical.
        if grid.face_count == 6:
            result, cg2d_iters, cg2d_residual = self._step_cube(
                state, model, bundle, time_seconds=plan.times.current,
            )
            return StepOutcome(
                state=result,
                diagnostics=StepDiagnostics(
                    cg2d_iters=cg2d_iters, cg2d_residual=cg2d_residual,
                ),
                retained=EMPTY_MAPPING,
            )

        boundary, enforce_velocity, enforce_boundary_tracers, bounded, eta_h, first_step = (
            self._prepare_regional_entry(
                state=state, model=model, grid=grid, plan=plan, dt_tracer=dt_tracer, dt_freesurf=dt_freesurf, hydro_history=hydro_history,
                boundary_evaluation=boundary_evaluation,
            )
        )

        bounded, freezing_flux, empmr_mass_flux, mass_to_r_unit, pmep_r, kpp_forcing, kpp, mixing = (
            self._prepare_step_forcing(bounded=bounded, model=model, grid=grid, plan=plan, dt_tracer=dt_tracer, first_step=first_step)
        )
        kpp_fields = kpp.fields if kpp is not None else None
        (evaluation_inputs, momentum_w, resolved_3d_coriolis, assembly_policy,
         kpp_ab_names, kpp_post_names, defer_hpg, momentum_inputs) = (
            self._prepare_step_tendency_inputs(bounded=bounded, model=model, grid=grid, hydro_history=hydro_history, pmep_r=pmep_r, kpp_forcing=kpp_forcing, kpp=kpp, mixing=mixing)
        )
        # Keep the evaluator's graph breaks at the program call site.
        ab2_tendency, post_ab2_tendency, diagnostics = self._evaluate_slow_tendency_phase(
            model, bounded, kpp=kpp_fields, assembly_policy=assembly_policy, inputs=momentum_inputs,
        )
        ab2_tendency, kpp_nonlocal = self._complete_step_tendencies(
            bounded=bounded, model=model, grid=grid, hydro_history=hydro_history, kpp=kpp, kpp_fields=kpp_fields, dt_mom=dt_mom, ab2_tendency=ab2_tendency,
            diagnostics=diagnostics, kpp_ab_names=kpp_ab_names,
        )

        tend_u, tend_v, tend_temp, tend_salt, next_ab_history = self._extrapolate_step_ab(ab2_tendency=ab2_tendency, ab_history=ab_history)
        tend_u, tend_v, tend_temp, tend_salt, recip_hfac_new = self._prepare_momentum_prediction(
            bounded=bounded, model=model, grid=grid, tend_u=tend_u, tend_v=tend_v, tend_temp=tend_temp, tend_salt=tend_salt,
            post_ab2_tendency=post_ab2_tendency, defer_hpg=defer_hpg, diagnostics=diagnostics, evaluation_inputs=evaluation_inputs,
            dt_mom=dt_mom, ab2_tendency=ab2_tendency,
        )

        # 2. Predict u* = u^n + dt·(AB2_tend - (1-α)·g·∇η^n) then IMPLDIFF.
        #    MITgcm: gUdPx = -phFac*dPhiHydX - psFac*phiSurfX
        #    where psFac = pfFacMom * (1 - implicSurfPress).
        #    Unified with the cube step body in ImplicitFreeSurfaceStepper._predict_velocity;
        #    regional passes u_mask/v_mask=None (segment does no
        #    masking here — the has_3d_masks block below owns that) and reads its
        #    own step-local KPP fields, also consumed by the tracer block below.
        # Keep the predictor here to preserve its grid-cache graph break.
        if self.mom_stepping:
            u_star, v_star = self._predict_velocity(
                bounded.u.data, bounded.v.data, bounded.eta.data, tend_u, tend_v, grid, alpha=alpha, dt_mom=dt_mom,
                field_viscosity=mixing if self.kpp_enabled or self.vertical_mixing_scheme != "none" else None,
                dtype=bounded.u.data.dtype, device=bounded.u.data.device, momentum_w=momentum_w,
            )
        else:
            # momStepping=.FALSE. skips DYNAMICS (forward_step.F:786-795): uVel
            # and vVel are not advanced.
            u_star, v_star = bounded.u.data, bounded.v.data
        if ckpt_dump.ENABLED:
            record_checkpoint("ustar")
            record_checkpoint("implicit")

        # Tracer predictor (non-staggered inline advance), placed *after* the
        # velocity segment, which fuses predict+PG+IMPLDIFF into one call.  No shared read/write state with the segment —
        # this block reads KPP/temp/salt and writes temp_new/salt_new; the
        # segment reads eta/u/v/tendencies and writes u_star/v_star.
        if self.stagger_tracer_stepping:
            # Tracers stay at T^n/S^n — will be advanced after velocity correction
            temp_new = bounded.temp.data
            salt_new = bounded.salt.data
        else:
            temp_new, salt_new = self._advance_synchronous_tracers(
                bounded=bounded, grid=grid, dt_tracer=dt_tracer, tend_temp=tend_temp, tend_salt=tend_salt, recip_hfac_new=recip_hfac_new,
                kpp=kpp, kpp_fields=kpp_fields, mixing=mixing, kpp_post_names=kpp_post_names, temperature_advection_scheme=temperature_advection_scheme,
                salinity_advection_scheme=salinity_advection_scheme, multi_dim_advection=multi_dim_advection,
                gm_contributions=self._gm_kwz_contributions(
                    diagnostics, grid, dtype=tend_temp.dtype, device=tend_temp.device),
            )
            temp_new, salt_new = self._hold_unstepped_tracers(
                bounded=bounded, temp_new=temp_new, salt_new=salt_new,
            )

        # The predictors and AB history have consumed the momentum-phase
        # tendencies. Drop the containers and their local tensor aliases before
        # the CG2D graph break so Dynamo need not return their unused fields.
        # Staggered tracer stepping creates its own tendencies after the solve;
        # autograd retains any tensors needed by the existing predictor tape.
        del ab2_tendency, post_ab2_tendency, tend_u, tend_v, tend_temp, tend_salt

        # Keep c_mask live across CG2D to preserve graph input and output order.
        u_star, v_star, temp_new, salt_new, u_mask, v_mask, c_mask = self._mask_predictors(
            grid=grid, u_star=u_star, v_star=v_star, temp_new=temp_new, salt_new=salt_new,
        )

        provider_active = self.external_pnh_provider is not None
        internal_nh_active = self.nonhydrostatic.nonhydrostatic and not provider_active
        w_vel_new_mit_for_cg3d: torch.Tensor | None = None
        phi_nh_history = None
        phi_nh_prev = None
        gw_nm1_history = None

        u_star, v_star, temp_new, salt_new = self._apply_predictor_boundaries(
            state=state, grid=grid, bounded=bounded, boundary=boundary, enforce_velocity=enforce_velocity,
            enforce_boundary_tracers=enforce_boundary_tracers, u_star=u_star, v_star=v_star, temp_new=temp_new, salt_new=salt_new,
        )

        if internal_nh_active:
            w_vel_mit, phi_nh_history = self._prepare_vertical_momentum_state(bounded=bounded, grid=grid, u_star=u_star)
            if phi_nh_history is None:
                # Without this torch call Dynamo skips this frame and traces 84 graphs instead of 48.
                phi_nh_history = torch.zeros_like(temp_new)
            phi_nh_prev, gw_prev = self._prepare_vertical_momentum_history(phi_nh_history=phi_nh_history, temp_new=temp_new)
            w_vel_new_mit_for_cg3d, gw_nm1_history = self._predict_vertical_momentum(
                bounded, w_vel_mit=w_vel_mit, phi_nh_prev=phi_nh_prev, gw_prev=gw_prev, dt_mom=dt_mom, grid=grid, model=model,
                diagnostics=diagnostics, kpp_fields=kpp_fields, resolved_3d_coriolis=resolved_3d_coriolis,
            )
            del w_vel_mit, gw_prev

        (div_transport_star, layer_thickness, hFacW, hFacS, etaH_prev,
         surface_mask, maskInC_3d, empmr_mass_flux) = self._prepare_surface_transport(
            grid=grid, u_star=u_star, v_star=v_star, bounded=bounded, eta_h=eta_h, hydro_history=hydro_history,
            empmr_mass_flux=empmr_mass_flux,
        )

        old_free_surf_term = internal_nh_active and not self.exact_conserv
        nh_surface_source, eta_source = self._prepare_surface_elevation_source(
            grid=grid, etaH_prev=etaH_prev, old_free_surf_term=old_free_surf_term, phi_nh_prev=phi_nh_prev,
        )
        # 6. Solve for η^{n+1}
        #    x0 policy: η·maskInC when OBs are open, else *raw* η — the
        #    regional dry/OB cells deliberately keep η (MITgcm update_etah.F),
        #    unlike the cube's η·surf.  The three states do not converge, so
        #    x0_mask is passed explicitly rather than defaulted.
        # The tolerance mode travels as an explicit argument: the
        # adapter passes ``self.cg_tolerance_mode`` to ``cg2d_solve``.
        # Keep CG2D here to preserve the solver graph breaks.
        if self.mom_stepping:
            eta_new, cg2d_iters, cg2d_residual = self._solve_free_surface(
                bounded.eta.data, etaH_prev, div_transport_star, grid, dt_mom=dt_mom, dt_freesurf=dt_freesurf, beta=beta,
                dtype=bounded.eta.data.dtype, device=bounded.eta.data.device, surface_mask=surface_mask, mask_in_c=maskInC_3d,
                x0_mask=maskInC_3d, use_real_freshwater_flux=self.use_real_freshwater_flux, empmr_mass_flux=empmr_mass_flux, free_surf_fac=1.0,
                mass_to_r_unit=mass_to_r_unit, eta_source=eta_source,
            )
            eta_n = self._retain_surface_elevation(eta_new=eta_new)
            u_new, v_new = self._correct_barotropic_velocity(
                eta_new=eta_new, u_star=u_star, v_star=v_star, grid=grid, alpha=alpha, dt_mom=dt_mom, surface_mask=surface_mask,
            )
        else:
            # SOLVE_FOR_PRESSURE and MOMENTUM_CORRECTION_STEP run only under
            # momStepping (forward_step.F:897-903, :911-917): etaN keeps its
            # value and no CG2D solve is reported (StepDiagnostics: None).
            eta_new = bounded.eta.data
            eta_n = self._retain_surface_elevation(eta_new=eta_new)
            cg2d_iters = cg2d_residual = None
            u_new, v_new = u_star, v_star

        dphi_nh = None
        nh_iters = None
        nh_first_residual = None
        nh_last_residual = None
        if internal_nh_active:
            (cg3d_b, phi_nh_init_masked, nh_mask,
             operator, preconditioner) = self._prepare_internal_pressure(
                u_star=u_star, v_star=v_star, grid=grid, model=model, bundle=bundle, dt_mom=dt_mom, dt_freesurf=dt_freesurf,
                temp_new=temp_new, phi_nh_history=phi_nh_history, eta_new=eta_new, etaH_prev=etaH_prev,
                w_vel_new_mit_for_cg3d=w_vel_new_mit_for_cg3d, old_free_surf_term=old_free_surf_term, nh_surface_source=nh_surface_source,
            )
            # Keep the CG3D disable boundary at the program call site.
            (
                phi_nh_new, phi_nh_history, dphi_nh,
                nh_iters, nh_first_residual, nh_last_residual,
            ) = ifs_nh.solve_internal_pressure(
                cg3d_b=cg3d_b, phi_nh_init_masked=phi_nh_init_masked,
                operator=operator, preconditioner=preconditioner, nh_mask=nh_mask,
                grid=grid, bundle=bundle, params=self.nonhydrostatic,
                tolerance_mode=self.cg3d_tolerance_mode, exact_conserv=self.exact_conserv,
            )
            u_new, v_new = self._complete_internal_pressure(
                u_new=u_new, v_new=v_new, grid=grid, dt_mom=dt_mom, phi_nh_new=phi_nh_new, surface_mask=surface_mask,
            )

        corrected_bounded, u_final, v_final = self._apply_corrected_boundaries(
            state=state, grid=grid, u_new=u_new, v_new=v_new, eta_new=eta_new, temp_new=temp_new, salt_new=salt_new,
            surface_mask=surface_mask, dt_mom=dt_mom, u_mask=u_mask, v_mask=v_mask, boundary=boundary, enforce_velocity=enforce_velocity,
        )

        eta_final, eta_H_new, eta_h_history, tracer_w = self._advance_continuity(
            state=state, grid=grid, bounded=bounded, corrected_bounded=corrected_bounded, u_final=u_final, v_final=v_final,
            eta_new=eta_new, etaH_prev=etaH_prev, beta=beta, dt_freesurf=dt_freesurf, layer_thickness=layer_thickness, hFacW=hFacW,
            hFacS=hFacS, empmr_mass_flux=empmr_mass_flux, mass_to_r_unit=mass_to_r_unit, boundary=boundary, hydro_history=hydro_history,
            implicit_vertical_advection_active=implicit_vertical_advection_active,
        )
        implicit_tracer_names, continuity_ab_names, continuity_post_names = self._select_tracer_terms()
        if self.stagger_tracer_stepping and (
            implicit_tracer_names
            or hasattr(model, 'evaluate_tracer_tendency_with_diagnostics')
        ):
            (tracer_temp_tendency, tracer_salt_tendency, kpp_nonlocal,
             tracer_nlfs, recip_hfac_surf_future, gm_contributions,
             tracer_w_advect) = self._evaluate_staggered_tracers(
                state=state, model=model, grid=grid, bounded=bounded, eta_final=eta_final, u_final=u_final, v_final=v_final,
                dt_tracer=dt_tracer, evaluation_inputs=evaluation_inputs, kpp=kpp, kpp_fields=kpp_fields, kpp_ab_names=kpp_ab_names, kpp_nonlocal=kpp_nonlocal,
                implicit_tracer_names=implicit_tracer_names, continuity_ab_names=continuity_ab_names, tracer_w=tracer_w,
            )
            tracer_temp_n = bounded.temp.data
            tracer_salt_n = bounded.salt.data
            temp_new, salt_new = self._prepare_staggered_tracer_prediction(
                tracer_temp_n=tracer_temp_n, tracer_salt_n=tracer_salt_n, dt_tracer=dt_tracer, tracer_temp_tendency=tracer_temp_tendency,
                tracer_salt_tendency=tracer_salt_tendency,
            )
            if self.stream_trace.hook is not None:
                # Held as the after_implicit station below (thermodynamics.F:317-337).
                held_temp, held_salt = self._hold_unstepped_tracers(
                    bounded=bounded, temp_new=temp_new, salt_new=salt_new,
                )
                self.stream_trace.emit_stream_trace(
                    "tracer.after_explicit",
                    temp=held_temp,
                    salt=held_salt,
                )
            if kpp_fields is not None and self.kpp_tracer_substeps > 1:
                if gm_contributions:
                    raise ValueError("GM/Redi Kwz does not reach the KPP tracer substeps")
                (substep_kpp_inputs, tracer_mask_current, tracer_recip_current,
                 recip_hfac_tracer_matrix) = self._prepare_subcycle_tracer_inputs(
                    bounded=bounded, model=model, grid=grid, kpp_forcing=kpp_forcing, freezing_flux=freezing_flux, temp_new=temp_new,
                    tracer_nlfs=tracer_nlfs, recip_hfac_surf_future=recip_hfac_surf_future,
                )
                # Keep the subcycle call here to preserve the column graph breaks.
                temp_new, salt_new = subcycle_kpp_tracers(
                    bounded.temp.data, bounded.salt.data, outer_u=bounded.u.data, outer_v=bounded.v.data,
                    temp_tendency=tracer_temp_tendency, salt_tendency=tracer_salt_tendency, tracer_w=tracer_w_advect,
                    kpp_inputs=substep_kpp_inputs, grid=grid, dt=dt_tracer, substeps=self.kpp_tracer_substeps,
                    temperature_implicit_advection=self.temperature_implicit_vertical_advection,
                    salinity_implicit_advection=self.salinity_implicit_vertical_advection,
                    temperature_advection_scheme=temperature_advection_scheme, salinity_advection_scheme=salinity_advection_scheme,
                    multi_dim_advection=multi_dim_advection, outer_nonlocal=kpp_nonlocal,
                    nonlocal_ab_names=kpp_ab_names, continuity_tracer_names=continuity_post_names, mask_c_current=tracer_mask_current,
                    recip_hfac_current_c=tracer_recip_current, recip_hfac=recip_hfac_tracer_matrix,
                )
            else:
                (_kpp_ts_enabled, _kpp_layer_thickness, _kpp_convective_implicit,
                 tracer_column_diffusion, tracer_kappa_rt, tracer_kappa_rs,
                 tracer_mask_current, tracer_recip_current,
                 recip_hfac_tracer_matrix) = self._prepare_tracer_column_inputs(
                    grid=grid, temp_new=temp_new, salt_new=salt_new, kpp=kpp, mixing=mixing, gm_contributions=gm_contributions, implicit_tracer_names=implicit_tracer_names,
                    tracer_w=tracer_w_advect, tracer_nlfs=tracer_nlfs, recip_hfac_surf_future=recip_hfac_surf_future,
                )
                nonlocal_kz = (self._kpp_nonlocal_diffusivities(kpp_fields, gm_contributions)
                               if _kpp_ts_enabled else (None, None))
                # Keep the convective column disable boundary at this call site.
                temp_new, salt_new = ifs_tracer.mix_and_solve_columns(
                    temp_new=temp_new, salt_new=salt_new, temp_reference=tracer_temp_n, salt_reference=tracer_salt_n,
                    kpp_enabled=_kpp_ts_enabled, kpp=kpp_fields, layer_thickness=_kpp_layer_thickness, grid=grid, dt_tracer=dt_tracer,
                    tracer_w=tracer_w_advect if implicit_tracer_names else None, temperature_implicit_advection="temp" in implicit_tracer_names,
                    salinity_implicit_advection="salt" in implicit_tracer_names,
                    temperature_advection_scheme=temperature_advection_scheme, salinity_advection_scheme=salinity_advection_scheme,
                    multi_dim_advection=multi_dim_advection, nonlocal_tracer_names=kpp_post_names,
                    continuity_names=continuity_post_names, convective_implicit=_kpp_convective_implicit,
                    implicit_tracer_diffusion=tracer_column_diffusion, kappa_rt=tracer_kappa_rt, kappa_rs=tracer_kappa_rs,
                    nonlocal_temp_diffusivity=nonlocal_kz[0], nonlocal_salt_diffusivity=nonlocal_kz[1],
                    recip_hfac=recip_hfac_tracer_matrix, mask_c_current=tracer_mask_current, recip_hfac_current_c=tracer_recip_current,
                    recip_hfac_continuity=recip_hfac_tracer_matrix,
                )
            temp_new, salt_new = self._hold_unstepped_tracers(
                bounded=bounded, temp_new=temp_new, salt_new=salt_new,
            )
            if self.stream_trace.hook is not None:
                self.stream_trace.emit_stream_trace("tracer.after_implicit", temp=temp_new, salt=salt_new)
            if grid.masks.has_3d_masks:
                # Preserve the original c_mask assignment and its cross-CG2D lifetime.
                c_mask = grid.masks.center_float(dtype=temp_new.dtype, device=temp_new.device)
                temp_new, salt_new = ifs_tracer.mask_tracer_pair(temp_new, salt_new, mask_c=c_mask)
            _dist_trace("staggered tracer update done")
        else:
            # Non-staggered path: tracers were already advanced above
            temp_new = corrected_bounded.temp.data
            salt_new = corrected_bounded.salt.data

        result, w_next_mit = self._finalize_regional_state(
            state=state, model=model, grid=grid, corrected_bounded=corrected_bounded, u_final=u_final, v_final=v_final,
            eta_final=eta_final, temp_new=temp_new, salt_new=salt_new, enforce_velocity=enforce_velocity,
            enforce_boundary_tracers=enforce_boundary_tracers, internal_nh_active=internal_nh_active, phi_nh_history=phi_nh_history,
            gw_nm1_history=gw_nm1_history, implicit_vertical_advection_active=implicit_vertical_advection_active, tracer_w=tracer_w,
            hydro_history=hydro_history, eta_h_history=eta_h_history,
        )
        return self._publish_step_outcome(
            result=result, ab_history=ab_history, hydro_history=hydro_history, next_ab_history=next_ab_history,
            eta_H_new=eta_H_new, w_next_mit=w_next_mit, cg2d_iters=cg2d_iters, cg2d_residual=cg2d_residual, eta_n=eta_n, dphi_nh=dphi_nh,
            nh_iters=nh_iters, nh_first_residual=nh_first_residual, nh_last_residual=nh_last_residual,
        )
