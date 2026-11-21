"""The shared mapping tables of input/data and input/eedata."""

from __future__ import annotations

from experiments.mitcases.code_inventory import MAP
from locomgcm.dynamics.vector_invariant_momentum import KE_SCHEMES, VORT_SCHEMES

from .core import (
    _ASSEMBLY,
    _IC_SOURCE_NAME,
)
from .rules import (
    Rule,
    _defer,
    _false_or_gap,
    _forcing_binary_input,
    _gap,
    _grid_switch,
    _horizontal_spacing,
    _ic_binary_input,
    _map,
    _map_both,
    _origin,
    _skip,
    _spacing_file,
    _stage,
    _unread_binary_input,
    _zero_or_gap,
)
from .convert import (
    _MIT_DEFAULT_TREF,
    _cd_parameter,
    _clim_file,
    _clim_relax_tau,
    _f_bool_from_int,
    _f_buoyancy,
    _f_cadj_freq,
    _f_celsius2k,
    _f_cg2d_min_iters_nsa,
    _f_cg2d_target_res_w_unit,
    _f_cg2d_target_residual,
    _f_cg2d_use_min_res_sol,
    _f_cg3d_target_residual,
    _f_convert_fw2salt,
    _f_delrc,
    _f_do_ab_on_gt_gs,
    _f_dt_alias,
    _f_dump_freq,
    _f_end_time,
    _f_eos,
    _f_free_surface,
    _f_gbaro,
    _f_grid_cap,
    _f_implicit_visc,
    _f_ivdc_kappa,
    _f_lat_band_clim_relax,
    _f_layers,
    _f_leith,
    _f_mit_start,
    _f_monitor_freq,
    _f_radius_from_horiz_grid,
    _f_real_freshwater_flux,
    _f_rigid_lid,
    _f_rotation_period,
    _f_rsphere,
    _f_sadourny_coriolis,
    _f_select_bot_drag_quadr,
    _f_selectp_in_eos_zc,
    _f_sref,
    _f_tracer_coefficient,
    _f_use_cd_scheme,
    _f_use_full_leith,
    _f_use_nsa_cg_solver,
    _f_use_pickup_before_c54,
    _f_vector_invariant_scheme,
    _f_visca4,
    _fill_ref,
    _matches,
    _tracer_adv_rule,
    _vertical_adv_rule,
    _vorticity_reconstruction_switch,
)

# --------------------------------------------------------------------------
# THE SHARED MAPPING TABLE -- ``input/data``
# --------------------------------------------------------------------------

DATA_RULES: dict[tuple[str, str], Rule] = {
    # ---- PARM01: continuous equation parameters -------------------------
    ("PARM01", "eostype"): _map(
        "model.dynamics <- eos_type", "MIT eosType name -> EOS_TYPES module", _f_eos
    ),
    ("PARM01", "readbinaryprec"): _map(
        "case.extras.mitcase.read_binary_prec",
        "bits -> big-endian dtype ('>f4'/'>f8'); consumed by binary ingest",
    ),
    ("PARM01", "writebinaryprec"): _map("case.extras.mitcase.write_binary_prec"),
    ("PARM01", "tref"): _map(
        f"sources.{_IC_SOURCE_NAME}.t_ref",
        "tRef filled down to Nr with the last given level (load_ref_files.F:45-53)",
        lambda v, ctx: {
            f"sources.{_IC_SOURCE_NAME}.t_ref": _fill_ref(v, ctx.nz, _MIT_DEFAULT_TREF, "tref")
        },
    ),
    ("PARM01", "sref"): _map(
        f"sources.{_IC_SOURCE_NAME}.s_ref",
        "sRef filled down to Nr with the last given level (load_ref_files.F:76-84)",
        _f_sref,
    ),
    ("PARM01", "treffile"): _gap("tRef read from file"),
    ("PARM01", "rhoreffile"): _gap("reference density profile read from file"),
    ("PARM01", "implicitfreesurface"): _map(
        "physics", "logical gate", _f_free_surface
    ),
    ("PARM01", "rigidlid"): _map("physics", "logical gate", _f_rigid_lid),
    ("PARM01", "viscah"): _map("physics.viscosity"),
    ("PARM01", "viscaz"): _map("physics.implicit_viscosity"),
    ("PARM01", "viscar"): _map("physics.implicit_viscosity"),
    ("PARM01", "diffkht"): _map("physics.diffusivity"),
    # The horizontal tracer diffusivities, temperature and salinity apart:
    # GAD_CALC_RHS takes diffKhT, diffK4T for temperature and diffKhS, diffK4S for
    # salinity (temp_integrate.F:341, :356; salt_integrate.F:339, :354), each 0 when
    # omitted (set_defaults.F:153-156, read at ini_parms.F:218).  The assembly
    # resolves the pair of every stepped tracer (_horizontal_tracer_diffusion).
    ("PARM01", "diffkhs"): _map(
        "model.dynamics.horizontal_tracer_diffusion.diffusivity (salt)",
        "identity, m2 s-1 (zero: no-op, the MIT default)",
        _f_tracer_coefficient("diffusivity_kh_salt", "diffKhS"),
    ),
    # The vertical salt diffusivities.  The vertical diffusivity is the single
    # physics.tracer.diffusivity_kr (config/_schema.py:373), one value for every
    # tracer (vertical_diffusion.py: one vertical_mixing consumer, one stepper
    # tracer_vertical_diffusivity_kr), so a salt value is admitted only when it
    # equals the temperature value MITgcm uses.  An unset vertical T diffusivity is
    # diffKrTDefault = 0 (ini_parms.F:570, :583-584, set_defaults.F:157); a set
    # diffKzS/diffKrS is the salt one (:602, :622-626), and only with no salt key set
    # does salt take the T profile (:626-632), which needs no row.  MITgcm stops on
    # z and r keys mixed (:746-755).
    ("PARM01", "diffkzt"): _map("physics.tracer.diffusivity_kr"),
    ("PARM01", "diffkrt"): _map("physics.tracer.diffusivity_kr"),
    ("PARM01", "diffkzs"): Rule(
        MAP,
        convert="whitelisted when equal to diffKzT (default 0)",
        fn=_matches(
            "PARM01", "diffkzt", "diffKzT", "LOCOMgcm carries one vertical tracer diffusivity",
            peer_default=0.0,
        ),
    ),
    ("PARM01", "diffkrs"): Rule(
        MAP,
        convert="whitelisted when equal to diffKrT (default 0)",
        fn=_matches(
            "PARM01", "diffkrt", "diffKrT", "LOCOMgcm carries one vertical tracer diffusivity",
            peer_default=0.0,
        ),
    ),
    ("PARM01", "gravity"): _map("model.stepper.kwargs.gravity"),
    ("PARM01", "gbaro"): _map("physics", "whitelisted when equal to gravity", _f_gbaro),
    ("PARM01", "rhoconst"): _stage(
        "reference_density", "rhoConst -> the assembly's Boussinesq reference density "
        "(recip_rhoConst, ini_parms.F:763); unset is rhoNil (:483)"
    ),
    ("PARM01", "rhonil"): _stage(
        "rho_nil", "rhoNil -> the linear EOS reference_density and the pressure gradient's "
        "rho_nil (find_rho.F:101-108)"
    ),
    ("PARM01", "no_slip_sides"): _map("physics.no_slip_sides"),
    ("PARM01", "no_slip_bottom"): _stage(
        "no_slip_bottom", "bottom_drag.no_slip_bottom (mom_u_botdrag_coeff.F)"
    ),
    ("PARM01", "implicitdiffusion"): _map("physics.tracer.implicit_vertical_diffusion"),
    ("PARM01", "implicitviscosity"): _map(
        "physics.implicit_vertical_viscosity",
        "MIT implicitViscosity gate, separate from viscAr / viscAz",
        _f_implicit_visc,
    ),
    # Both hFac floors live in case.extras: that is the location the grid
    # builders read (experiments/cases/_runtime.py:649-650,
    # model_input.py:786-787) and the one legacy_bridge.py:429-444 claims.
    # physics.hfac_min_dz exists in the schema but has no consumer, and
    # _assert_yaml_complete rejects it outright.  An absent hFacMin is written
    # by _apply_mit_hfac_min_default.
    ("PARM01", "hfacmin"): _map("case.extras.hfac_min"),
    ("PARM01", "hfacmindr"): _map("case.extras.hfac_min_dz"),
    ("PARM01", "hfacmindz"): _map("case.extras.hfac_min_dz"),
    ("PARM01", "hfacinf"): _map("model.stepper.kwargs.hfac_inf"),
    ("PARM01", "hfacsup"): _skip(
        "MITgcm applies hFacSup only on the r* branch (calc_r_star.F:194,249); "
        "with select_rStar=0 the whole tree references it just to print a "
        "WARNING (calc_surf_dr.F:126-135), so it has no effect on the state"
    ),
    ("PARM01", "staggertimestep"): _map("model.stepper.kwargs.stagger_tracer_stepping"),
    ("PARM01", "exactconserv"): _map("model.stepper.kwargs.exact_conserv"),
    ("PARM01", "implicsurfpress"): _map("model.stepper.kwargs.implicit_surf_press"),
    ("PARM01", "implicdiv2dflow"): _map("model.stepper.kwargs.implicit_div_2d_flow"),
    ("PARM01", "implicitintgravwave"): _map("model.stepper.kwargs.implicit_int_grav_wave"),
    ("PARM01", "allowfreezing"): _map("model.stepper.kwargs.allow_freezing"),
    # Both targets, deliberately.  physics.* is the declaration the registry
    # record and the WebUI read; model.stepper.kwargs.* is what
    # _instantiate_stepper actually passes to ImplicitFreeSurfaceStepper.
    # Writing only physics.* left internal_wave declaring nonlinFreeSurf=3 while
    # the stepper ran level 0 -- a target with no consumer is not a translation

    ("PARM01", "nonlinfreesurf"): _map_both(
        "physics.nonlin_free_surf", "model.stepper.kwargs.nonlin_free_surf"
    ),
    ("PARM01", "select_rstar"): _map_both(
        "physics.free_surface.select_r_star", "model.stepper.kwargs.select_r_star"
    ),
    ("PARM01", "selectkescheme"): _map(
        "model.dynamics.vector_invariant_momentum_advection.select_ke_scheme",
        "MIT code in KE_SCHEMES (vector-invariant only); also bottom_drag.ke_scheme "
        "and physics.ke_scheme",
        _f_vector_invariant_scheme(
            "selectKEscheme", KE_SCHEMES, "select_ke_scheme", declared="KE_SCHEMES"
        ),
    ),
    ("PARM01", "nonhydrostatic"): _map("physics.nonhydrostatic.enabled"),
    ("PARM01", "select3dcorischeme"): _stage(
        "select_3d_cori_scheme",
        "resolved with use3dCoriolis and the effective Coriolis map",
    ),
    ("PARM01", "use3dcoriolis"): _stage(
        "use_3d_coriolis",
        "backward-compatible selector used only when select3dCoriScheme is unset",
    ),
    ("PARM01", "tempadvscheme"): _tracer_adv_rule("temp"),
    ("PARM01", "saltadvscheme"): _tracer_adv_rule("salt"),
    ("PARM01", "tempvertadvscheme"): _vertical_adv_rule("temp"),
    ("PARM01", "saltvertadvscheme"): _vertical_adv_rule("salt"),
    ("PARM01", "multidimadvection"): _map("advection.multi_dim"),
    ("PARM01", "tempimplvertadv"): _map("physics.implicit_vertical_advection.temperature"),
    ("PARM01", "saltimplvertadv"): _map("physics.implicit_vertical_advection.salinity"),
    ("PARM01", "viscc2leith"): _map(
        "physics.visc_c2_leith",
        "also sets physics.viscosity_scheme=leith_smag",
        _f_leith("physics.visc_c2_leith"),
    ),
    ("PARM01", "viscc2smag"): _map(
        "physics.visc_c2_smag",
        "also sets physics.viscosity_scheme=leith_smag",
        _f_leith("physics.visc_c2_smag"),
    ),
    ("PARM01", "visca4"): _map(
        "model.dynamics.horizontal_viscosity.viscosity_a4",
        "identity (zero: no-op, the module default)",
        _f_visca4,
    ),
    # GAD_CALC_RHS takes diffK4T for temperature, diffK4S for salinity
    # (temp_integrate.F:341, :356; salt_integrate.F:339, :354): with diffK4 /= 0 it
    # forms del2 by GAD_GRAD_X/Y and GAD_DEL2 (gad_calc_rhs.F:229-245) and adds the
    # GAD_BIHARM_X/Y flux to the harmonic one (:339-340, :468-469) =
    # HorizontalTracerDiffusion._diffuse_biharmonic.  Defaults 0
    # (set_defaults.F:155-156); the assembly resolves each tracer's coefficient.
    ("PARM01", "diffk4t"): _map(
        "model.dynamics.horizontal_tracer_diffusion.diffusivity_k4 (temp)",
        "identity, m4 s-1 (zero: no-op, the MIT default)",
        _f_tracer_coefficient("diffusivity_k4_temp", "diffK4T"),
    ),
    ("PARM01", "diffk4s"): _map(
        "model.dynamics.horizontal_tracer_diffusion.diffusivity_k4 (salt)",
        "identity, m4 s-1 (zero: no-op, the MIT default)",
        _f_tracer_coefficient("diffusivity_k4_salt", "diffK4S"),
    ),
    ("PARM01", "ivdc_kappa"): _map(
        "model.stepper.kwargs.{ivdc_kappa,implicit_tracer_diffusion}",
        "identity, m2 s-1 (zero: no-op); outside the verbatim mode the case-scope rule",
        _f_ivdc_kappa,
    ),
    # The rest of MOM_CALC_VISC's variable viscosity: each keyword is one
    # HorizontalViscosity field, read only by the leith_smag / leith_smag_flux
    # branches, as MITgcm reads it only in MOM_CALC_VISC, which it calls only
    # with useVariableVisc (set_parms.F:132-138; mom_vecinv.F:373,
    # mom_fluxform.F:453).  A nonzero coefficient selects that scheme the way
    # viscC2leith does.  Factors: leith4fac 0.125 (c/pi)**3, or 0.015625
    # (c/pi)**6 with useFullLeith, smag4fac 0.125 (c/pi)**2 (mom_calc_visc.F:
    # 196-218) = _biharmonic_coeffs_from_measures; leithD2fac (c/pi)**3 or **6
    # = _harmonic_from_measures via _leith_contrib.  The biharmonic stencil is
    # MOM_VI_DEL2UV then MOM_VI_HDISSIP's del^4 block (mom_vecinv.F:400-406,
    # :429-434) = _biharmonic_dissipation under leith_smag, and del2u fed to the
    # viscous fluxes (mom_fluxform.F:601-669) = _flux_form_biharmonic under
    # leith_smag_flux.
    ("PARM01", "viscc4leith"): _map(
        "model.dynamics.horizontal_viscosity.visc_c4_leith",
        "identity; also sets physics.viscosity_scheme=leith_smag or leith_smag_flux",
        _f_leith(f"{_ASSEMBLY}.visc_c4_leith"),
    ),
    ("PARM01", "viscc4leithd"): _map(
        "model.dynamics.horizontal_viscosity.visc_c4_leith_d",
        "identity; also sets physics.viscosity_scheme=leith_smag or leith_smag_flux",
        _f_leith(f"{_ASSEMBLY}.visc_c4_leith_d"),
    ),
    ("PARM01", "viscc4smag"): _map(
        "model.dynamics.horizontal_viscosity.visc_c4_smag",
        "identity; also sets physics.viscosity_scheme=leith_smag or leith_smag_flux",
        _f_leith(f"{_ASSEMBLY}.visc_c4_smag"),
    ),
    ("PARM01", "viscc2leithd"): _map(
        "model.dynamics.horizontal_viscosity.visc_c2_leith_d",
        "identity; also sets physics.viscosity_scheme=leith_smag or leith_smag_flux",
        _f_leith(f"{_ASSEMBLY}.visc_c2_leith_d"),
    ),
    # QG Leith adds the vortex-stretching gradient (mom_calc_visc.F:318-362,
    # under ALLOW_LEITH_QG); LOCOMgcm has no stretching term.
    ("PARM01", "viscc2leithqg"): _zero_or_gap("QG Leith viscosity (viscC2LeithQG)"),
    ("PARM01", "viscahgridmax"): _map(
        "model.dynamics.horizontal_viscosity.visc_ah_grid_max",
        "identity when > 0 (with dt = deltaTmom), else capability gap",
        _f_grid_cap("visc_ah_grid_max", "viscAhGridMax"),
    ),
    ("PARM01", "visca4gridmax"): _map(
        "model.dynamics.horizontal_viscosity.visc_a4_grid_max",
        "identity when > 0 (with dt = deltaTmom), else capability gap",
        _f_grid_cap("visc_a4_grid_max", "viscA4GridMax"),
    ),
    # .TRUE. takes L2 = rA, rAz (mom_init_fixed.F:94-129); LOCOMgcm's lengths are
    # only the .FALSE. form 2/(dx**-2 + dy**-2) (_harmonic_viscosity_length_scales).
    ("PARM01", "useareavisclength"): _false_or_gap("area-based viscosity length scale"),
    # The L2-norm branch with the **6 factors (mom_calc_visc.F:204-211, :406-453,
    # :561-605) = _leith_viscosity_contribution(use_full_leith=True).
    # Also read by GMREDI_CALC_QGLEITH (gmredi_calc_qgleith.F:68, :223) under
    # GM_useLeithQG; useGMRedi is a capability gap (PACKAGE_RULES).
    ("PARM01", "usefullleith"): _map(
        "model.dynamics <- use_full_leith",
        "identity -> horizontal_viscosity.use_full_leith; skip with useVariableVisc false",
        _f_use_full_leith,
    ),
    # .TRUE. calls MOM_HDISSIP, the tension-strain stencil (mom_vecinv.F:419-427);
    # LOCOMgcm's horizontal_viscosity has only the div-vort and flux-divergence ones.
    ("PARM01", "usestraintensionvisc"): _false_or_gap("strain-tension viscosity form"),
    ("PARM01", "usesmag3d"): _false_or_gap("3-D Smagorinsky viscosity"),
    ("PARM01", "smag3d_coeff"): _zero_or_gap("3-D Smagorinsky coefficient"),
    ("PARM01", "cospower"): _gap("cos-power meridional viscosity scaling (cosPower)"),
    ("PARM01", "usecdscheme"): _map(
        "model.dynamics.cd_scheme",
        ".TRUE. -> the cd_scheme module, no C-grid Coriolis; outside the verbatim mode the case-scope rule",
        _f_use_cd_scheme,
    ),
    ("PARM01", "useabsvorticity"): _false_or_gap("absolute-vorticity vector-invariant form"),
    ("PARM01", "sadournycoriolis"): _map(
        "model.dynamics.vector_invariant_momentum_advection.select_vort_scheme",
        ".TRUE. -> selectVortScheme 2 when unset (ini_parms.F:694-704)",
        _f_sadourny_coriolis,
    ),
    ("PARM01", "highordervorticity"): _vorticity_reconstruction_switch(
        "highOrderVorticity", "high_order_vorticity", "upwindVorticity"
    ),
    ("PARM01", "upwindvorticity"): _vorticity_reconstruction_switch(
        "upwindVorticity", "upwind_vorticity", "highOrderVorticity"
    ),
    # .TRUE. multiplies the vorticity term by the wet-point factor of
    # mom_vi_u_coriolis.F:179-190, which LOCOMgcm has not ported.
    ("PARM01", "usejamartmomadv"): _false_or_gap("useJamartMomAdv wet-point vorticity factor"),
    ("PARM01", "selectvortscheme"): _map(
        "model.dynamics.vector_invariant_momentum_advection.select_vort_scheme",
        "MIT code in VORT_SCHEMES (vector-invariant only)",
        _f_vector_invariant_scheme(
            "selectVortScheme", VORT_SCHEMES, "select_vort_scheme", declared="VORT_SCHEMES"
        ),
    ),
    # selectCoriScheme and its backward-compatible spelling useJamartWetPoints are
    # resolved into one code (_mit_cori_scheme, ini_parms.F:661-692), which the
    # momentum assembly writes to the module holding the Coriolis term.
    ("PARM01", "selectcorischeme"): _stage(
        "select_cori_scheme",
        "with useJamartWetPoints (ini_parms.F:661-692) -> select_cori_scheme of the Coriolis "
        "module (CORI_SCHEMES, FLUX_CORI_SCHEMES)",
    ),
    ("PARM01", "selectcorimap"): _stage(
        "select_cori_map",
        "effective map and fCoriCos source (set_parms.F:73-85; ini_cori.F:55-100)",
    ),
    ("PARM01", "selectimplicitdrag"): _gap("implicit bottom drag selector"),
    ("PARM01", "selectnhfreesurf"): _gap("non-hydrostatic free-surface selector"),
    ("PARM01", "selectfindrosurf"): _gap("Ro_surf determination selector"),
    ("PARM01", "selectaddfluid"): _gap("addFluid (mass source) selector"),
    ("PARM01", "selectbotdragquadr"): _map(
        "model.dynamics.bottom_drag",
        "-1 and 0 match BottomDrag's one form, else capability gap",
        _f_select_bot_drag_quadr,
    ),
    ("PARM01", "selectp_ineos_zc"): _map(
        "model.dynamics EOS kwargs",
        "resolved (set_parms.F:275-298): 0 -> dataclass default false; 2 -> dynamic_pressure "
        "true; 1, 3 and MITgcm's stops -> gap",
        _f_selectp_in_eos_zc,
    ),
    # Read only by ini_parms.F:661-692 (no other reader in model/ or pkg/).
    ("PARM01", "usejamartwetpoints"): _stage(
        "use_jamart_wet_points",
        "selectCoriScheme = 1 when selectCoriScheme is unset (ini_parms.F:661-663), "
        "see selectCoriScheme",
    ),
    ("PARM01", "usenhmterms"): _false_or_gap("non-hydrostatic metric terms"),
    ("PARM01", "quasihydrostatic"): _false_or_gap("quasi-hydrostatic approximation"),
    ("PARM01", "doresethfactors"): _defer("per-step hFac reset (nonlinFreeSurf)"),
    ("PARM01", "uniformlin_phisurf"): _defer(
        "linear free-surface phiSurf uniformity option"
    ),
    ("PARM01", "momstepping"): _stage(
        "mom_stepping", "gates every momentum module (set_parms.F:88-95) and sets "
        "model.stepper.kwargs.mom_stepping (_wire_stepping_switches)"
    ),
    ("PARM01", "tempstepping"): _stage(
        "temp_stepping", "'temp' joins the prognostic tracer set; model.stepper.kwargs.temp_stepping"
    ),
    ("PARM01", "saltstepping"): _stage(
        "salt_stepping", "'salt' joins the prognostic tracer set; model.stepper.kwargs.salt_stepping"
    ),
    ("PARM01", "momadvection"): _stage(
        "mom_advection", "flux-form/vector-invariant bundle vs Coriolis-only"
    ),
    ("PARM01", "momviscosity"): _stage(
        "mom_viscosity", "gates the horizontal_viscosity entry"
    ),
    ("PARM01", "tempadvection"): _stage(
        "temp_advection", "gates the horizontal_tracer_advection entry"
    ),
    ("PARM01", "saltadvection"): _stage(
        "salt_advection", "gates salt in the horizontal_tracer_advection entry"
    ),
    ("PARM01", "vectorinvariantmomentum"): _stage(
        "vector_invariant",
        "selects vector_invariant_momentum_advection over flux_form_momentum",
    ),
    ("PARM01", "f0"): _stage("f0", "Coriolis map intercept (ini_cori.F:77)"),
    ("PARM01", "beta"): _stage("beta", "Coriolis map slope (ini_cori.F:77)"),
    ("PARM01", "fprime"): _stage(
        "f_prime", "constant-F/Beta fCoriCos value (ini_cori.F:63,79)"
    ),
    ("PARM01", "usecoriolis"): _stage(
        "use_coriolis", "gates the Coriolis term of the momentum assembly"
    ),
    ("PARM01", "rotationperiod"): _map(
        "physics.rotation_period",
        "identity, s (selectCoriMap != 2 or == ROTATION_PERIOD_EARTH: no-op)",
        _f_rotation_period,
    ),
    ("PARM01", "talpha"): _stage(
        "t_alpha", "linear-EOS thermal expansion (ini_eos.F:85)"
    ),
    ("PARM01", "sbeta"): _stage(
        "s_beta", "linear-EOS haline contraction (ini_eos.F:86)"
    ),
    ("PARM01", "bottomdraglinear"): _stage(
        "bottom_drag_linear", "bottom_drag.linear (mom_u_botdrag_coeff.F)"
    ),
    ("PARM01", "bottomdragquadratic"): _stage(
        "bottom_drag_quadratic", "bottom_drag.quadratic (mom_u_botdrag_coeff.F)"
    ),
    ("PARM01", "zroughbot"): _gap("bottom roughness length (zRoughBot)"),
    ("PARM01", "addfrictionheating"): _false_or_gap("frictional heating source term"),
    ("PARM01", "salt_staypositive"): _false_or_gap("positivity-preserving salt limiter"),
    ("PARM01", "selectpenetratingsw"): _stage(
        "select_penetrating_sw", "resolved with SHORTWAVE_HEATING for surface forcing and KPP"
    ),
    ("PARM01", "heatcapacity_cp"): _stage(
        "heat_capacity", "HeatCapacity_Cp -> wpflux_surface_forcing.heat_capacity"
    ),
    ("PARM01", "rhoconstfresh"): _stage(
        "rho_const_fresh", "rhoConstFresh -> wpflux_surface_forcing.rho_const_fresh"
    ),
    ("PARM01", "convertfw2salt"): Rule(
        MAP,
        target="model.dynamics <- convert_fw2salt",
        convert=(
            "uniform value -> wpflux_surface_forcing; -1 -> current local "
            "surface salinity"
        ),
        fn=_f_convert_fw2salt,
    ),
    ("PARM01", "userealfreshwaterflux"): Rule(
        MAP,
        target="model.stepper.kwargs.use_real_freshwater_flux",
        convert=(
            ".FALSE. -> no-op; .TRUE. -> integrated-eta CG2D RHS and "
            "exactConserv volume coupling (surface vertical-momentum effect omitted)"
        ),
        fn=_f_real_freshwater_flux,
    ),
    ("PARM01", "temp_evprrn"): _defer("evaporated-water temperature; forcing"),
    ("PARM01", "hmixcriteria"): _gap("mixed-layer-depth diagnostic criterion"),
    ("PARM01", "diffkrnrt"): _gap("depth-varying background T diffusivity (diffKrNrT)"),
    ("PARM01", "diffkrnrs"): _gap("depth-varying background S diffusivity (diffKrNrS)"),
    ("PARM01", "diffkrbl79surf"): _gap("Bryan-Lewis 1979 vertical diffusivity profile"),
    ("PARM01", "diffkrbl79deep"): _gap("Bryan-Lewis 1979 vertical diffusivity profile"),
    ("PARM01", "diffkrbl79scl"): _gap("Bryan-Lewis 1979 vertical diffusivity profile"),
    ("PARM01", "diffkrbl79ho"): _gap("Bryan-Lewis 1979 vertical diffusivity profile"),
    ("PARM01", "momtidalforcing"): _false_or_gap("tidal body forcing on momentum"),
    ("PARM01", "buoyancyrelation"): Rule(
        MAP, convert="'OCEANIC' only", fn=lambda v, ctx: _f_buoyancy(v, ctx)
    ),
    ("PARM01", "integr_geopot"): _gap("pressure-coordinate geopotential integration"),
    ("PARM01", "atm_cp"): _gap("atmospheric configuration (atm_Cp)"),
    ("PARM01", "atm_rq"): _gap("atmospheric configuration (atm_Rq)"),
    ("PARM01", "celsius2k"): _map(
        "locomgcm.dynamics.bulk_formulae.CEN2KEL",
        "matches when equal to CEN2KEL, else capability gap",
        _f_celsius2k,
    ),
    ("PARM01", "globalfiles"): _skip("MIT single-file I/O layout"),
    ("PARM01", "usesinglecpuio"): _skip("MIT I/O gather strategy"),
    ("PARM01", "plotlevel"): _skip("MIT print verbosity"),
    ("PARM01", "debuglevel"): _skip("MIT debug verbosity"),
    ("PARM01", "usepickupbeforec54"): _map(
        "initial_condition.mit_pickup",
        ".FALSE. -> field-list layout; .TRUE. -> unsupported at a pickup start (_f_mit_start)",
        _f_use_pickup_before_c54,
    ),
    ("PARM01", "smoothabsfuncrange"): _defer("adjoint abs() smoothing; adjoint axis"),
    # ---- PARM02: elliptic solver ---------------------------------------
    ("PARM02", "cg2dmaxiters"): _map("model.stepper.kwargs.cg_max_iters"),
    ("PARM02", "cg2dtargetresidual"): _map(
        "model.stepper.kwargs.{cg_tolerance,cg_tolerance_mode}",
        "dimensionless target after MIT RHS-max normalization",
        _f_cg2d_target_residual,
    ),
    ("PARM02", "cg3dmaxiters"): _map("physics.nonhydrostatic.cg3d_max_iters"),
    ("PARM02", "cg3dtargetresidual"): _map(
        "physics.nonhydrostatic.{cg3d_target_residual} + "
        "model.stepper.kwargs.{cg3d_tolerance_mode}",
        "identity",
        _f_cg3d_target_residual,
    ),
    ("PARM02", "cg3dtargetreswunit"): _map("physics.nonhydrostatic.cg3d_target_res_w_unit"),
    ("PARM02", "cg2dtargetreswunit"): _map(
        "model.stepper.kwargs.{cg_tolerance,cg_tolerance_mode}",
        "<= 0 -> the normalised test of cg2dTargetResidual; > 0 -> W m/s, mode mit_w_unit",
        _f_cg2d_target_res_w_unit,
    ),
    # nIterMin = cg2dUseMinResSol - 1 (solve_for_pressure.F:278): a value >= 1 keeps
    # the lowest-residual iterate (cg2d.F:148, :338-341) and returns it when the
    # solve ends above it (:358-364).  Absent, _apply_mit_cg2d_defaults resolves
    # MITgcm's conditional default (ini_parms.F:1585-1588).
    ("PARM02", "cg2duseminressol"): _map(
        "model.stepper.kwargs.cg_use_min_res_sol",
        "value >= 1 -> true (nIterMin = value - 1 >= 0)",
        _f_cg2d_use_min_res_sol,
    ),
    ("PARM02", "usensacgsolver"): Rule(
        MAP, convert="skip: CG2D_NSA iterates as CG2D apart from cg2dMinItersNSA",
        fn=_f_use_nsa_cg_solver,
    ),
    ("PARM02", "usesrcgsolver"): _skip("MIT CG solver variant"),
    ("PARM02", "cg2dminitersnsa"): Rule(
        MAP, convert="skip when zero or when CG2D_NSA is not selected; else capability gap",
        fn=_f_cg2d_min_iters_nsa,
    ),
    ("PARM02", "printresidualfreq"): _skip("MIT solver print cadence"),
    # ---- PARM03: time stepping -----------------------------------------
    ("PARM03", "deltat"): _map("time.dt"),
    ("PARM03", "deltatmom"): _map(
        "time.dt_mom",
        "also seeds time.dt when deltaT / deltaTClock are absent",
        _f_dt_alias("time.dt_mom", "deltatmom"),
    ),
    ("PARM03", "deltattracer"): _map(
        "time.dt_tracer",
        "also seeds time.dt when no earlier deltaT* is present",
        _f_dt_alias("time.dt_tracer", "deltattracer"),
    ),
    ("PARM03", "deltatfreesurf"): _map(
        "time.dt_freesurf",
        "also seeds time.dt when no earlier deltaT* is present",
        _f_dt_alias("time.dt_freesurf", "deltatfreesurf"),
    ),
    ("PARM03", "deltatclock"): _map(
        "time.dt_clock",
        "also seeds time.dt when deltaT is absent",
        _f_dt_alias("time.dt_clock", "deltatclock"),
    ),
    ("PARM03", "ntimesteps"): _map("time.steps"),
    ("PARM03", "endtime"): _map(
        "time.seconds", "endTime - startTime, when nTimeSteps is absent", _f_end_time
    ),
    # startTime, nIter0, baseTime and pickupSuff decide together whether INI_FIELDS reads the
    # IC files or a pickup (ini_fields.F:27-38, the clock of ini_parms.F:1121-1133); LOCOMgcm's
    # time zero is the resolved startTime, which the time-dependent sources carry.
    ("PARM03", "starttime"): _map(
        "initial_condition.mit_pickup", "IC start -> sources' start_time; pickup start -> "
        "mit_pickup (_f_mit_start)", _f_mit_start),
    ("PARM03", "niter0"): _map(
        "initial_condition.mit_pickup", "IC start -> sources' start_time; pickup start -> "
        "mit_pickup (_f_mit_start)", _f_mit_start),
    ("PARM03", "basetime"): _map(
        "initial_condition.mit_pickup", "IC start -> sources' start_time; pickup start -> "
        "mit_pickup (_f_mit_start)", _f_mit_start),
    ("PARM03", "pickupsuff"): _map(
        "initial_condition.mit_pickup", "IC start -> sources' start_time; pickup start -> "
        "mit_pickup (_f_mit_start)", _f_mit_start),
    ("PARM03", "rwsuffixtype"): _map(
        "initial_condition.mit_pickup", "0 -> I10.10 pickup suffix; else unsupported at a pickup "
        "start with blank pickupSuff (_f_mit_start)", _f_mit_start),
    ("PARM03", "abeps"): _map("model.stepper.kwargs.ab_eps"),
    ("PARM03", "alph_ab"): _gap("AB3 coefficient alph_AB (LOCOMgcm steps with AB2)"),
    ("PARM03", "beta_ab"): _gap("AB3 coefficient beta_AB (LOCOMgcm steps with AB2)"),
    ("PARM03", "momdissip_in_ab"): _map("model.stepper.kwargs.mom_dissip_in_ab"),
    ("PARM03", "momforcingoutab"): _map(
        "model.stepper.kwargs.mom_forcing_out_ab",
        "MIT integer switch -> bool",
        _f_bool_from_int("model.stepper.kwargs.mom_forcing_out_ab"),
    ),
    ("PARM03", "tracforcingoutab"): _stage(
        "trac_forcing_out_ab",
        "0 includes tracer forcing in AB; 1 applies it after AB",
    ),
    ("PARM03", "forcing_in_ab"): _stage(
        "forcing_in_ab",
        "sets unset momentum/tracer forcing order (ini_parms.F:1095-1101)",
    ),
    ("PARM03", "doab_ongtgs"): _map(
        "model.stepper.kwargs.{ab2_on_temperature_tendency,ab2_on_salinity_tendency}",
        "gad_init_fixed.F:143-163 per-tracer tendency gate; state AB -> capability gap",
        _f_do_ab_on_gt_gs,
    ),
    ("PARM03", "doab_ontands"): _false_or_gap(
        "AB on T/S fields has no LOCOMgcm counterpart"
    ),
    ("PARM03", "monitorfreq"): _map(
        "output.monitor_interval_steps", "monitorFreq / deltaT, rounded", _f_monitor_freq
    ),
    ("PARM03", "monitorselect"): _skip("MIT monitor field selection"),
    ("PARM03", "dumpfreq"): _map(
        "output.snapshot_interval_seconds", "seconds, verbatim", _f_dump_freq
    ),
    ("PARM03", "dumpinitandlast"): _skip("MIT dump cadence detail"),
    ("PARM03", "pchkptfreq"): _skip("MIT permanent-checkpoint cadence"),
    ("PARM03", "chkptfreq"): _skip("MIT rolling-checkpoint cadence"),
    ("PARM03", "writepickupatend"): _skip("MIT pickup write policy"),
    ("PARM03", "pickupstrictlymatch"): _skip(
        "with a field-list pickup it only decides whether a missing record stops the run "
        "(check_pickup.F:215-222) or zeroes its start-AB flag (:179-198), which "
        "pickup_ab_start reproduces from the records present (locomgcm/io/mit_pickup.py)"),
    ("PARM03", "diagfreq"): _skip("pkg/diagnostics cadence (governed by useDiagnostics)"),
    ("PARM03", "adjmonitorfreq"): _skip(
        "adjoint monitor cadence, no forward effect: forward_step.F:1151 also enters MONITOR when "
        "it is > 0, but MONITOR writes only on monitorFreq (pkg/monitor/monitor.F:48); the other "
        "reads are adjoint routines (monitor_ad.F:63, monitor_g.F:59, seaice_monitor_ad.F:61, "
        "ptracers_monitor_ad.F:60, exf_monitor_ad.F:61 via exf_readparms.F:305) and the pkg/autodiff "
        "start-up stop without ALLOW_AUTODIFF_MONITOR (autodiff_check.F:120)"
    ),
    ("PARM03", "adjdumpfreq"): _skip(
        "adjoint dump cadence, no forward effect: read only by adjoint and tangent-linear routines "
        "(pkg/autodiff/addummy_*.F, g_dummy_*.F, seaice_ad_dump.F:78, ptracers_ad_dump.F:69, "
        "exf_adjoint_snapshots_ad.F:78, exf_adjoint_snapshots__g.F:65), by write_grid.F:223 under "
        "ALLOW_ADJOINT_RUN or ALLOW_ADMTLM, and by the pkg/autodiff start-up stop without "
        "ALLOW_AUTODIFF_MONITOR (autodiff_check.F:114)"
    ),
    ("PARM03", "taucd"): _map(
        "model.dynamics.cd_scheme.tau_cd", "identity, s (0: deltaTMom, ini_parms.F:1116)",
        _cd_parameter("tau_cd", "C-D scheme timescale (governed by useCDscheme)"),
    ),
    ("PARM03", "rcd"): _map(
        "model.dynamics.cd_scheme.r_cd", "identity (negative: 1 - deltaTMom/tauCD, ini_parms.F:1117)",
        _cd_parameter("r_cd", "C-D scheme relaxation (read only with useCDscheme, ini_parms.F:1114-1118)"),
    ),
    ("PARM03", "epsab_cd"): _map(
        "model.dynamics.cd_scheme.eps_ab_cd", "identity (unset: abEps, ini_parms.F:1118)",
        _cd_parameter("eps_ab_cd", "C-D scheme AB epsilon (read only with useCDscheme, ini_parms.F:1114-1118)"),
    ),
    ("PARM03", "cadjfreq"): _map(
        "model.dynamics.convective_adjustment.c_adj_freq",
        "zero -> no-op; s, negative -> deltaTClock (ini_parms.F:1104-1106)",
        _f_cadj_freq,
    ),
    ("PARM03", "tauthetaclimrelax"): Rule(
        MAP,
        target="model.dynamics[surface_climate_relaxation].tau_theta",
        convert="identity, s; <= 0: no restoring (set_parms.F:250-253)",
        fn=_clim_relax_tau("theta"),
    ),
    ("PARM03", "tausaltclimrelax"): Rule(
        MAP,
        target="model.dynamics[surface_climate_relaxation].tau_salt",
        convert="identity, s; <= 0: no restoring (set_parms.F:250-253)",
        fn=_clim_relax_tau("salt"),
    ),
    ("PARM03", "latbandclimrelax"): Rule(
        MAP,
        target="model.dynamics[surface_climate_relaxation].lat_band",
        convert="identity, yC units (ini_forcing.F:50-61)",
        fn=_f_lat_band_clim_relax,
    ),
    ("PARM03", "externforcingcycle"): _stage(
        "extern_forcing_cycle", "seconds per surface-forcing cycle"
    ),
    ("PARM03", "externforcingperiod"): _stage(
        "extern_forcing_period", "seconds between surface-forcing records"
    ),
    ("PARM03", "periodicexternalforcing"): _stage(
        "periodic_external_forcing",
        "cycles the surface forcing file instead of holding record 1",
    ),
    # ---- PARM04: gridding ----------------------------------------------
    ("PARM04", "usingcartesiangrid"): _map(
        "case.extras.mitcase.grid_family", "using*Grid switch -> grid family", _grid_switch
    ),
    ("PARM04", "usingsphericalpolargrid"): _map(
        "case.extras.mitcase.grid_family", "using*Grid switch -> grid family", _grid_switch
    ),
    ("PARM04", "usingcurvilineargrid"): _map(
        "case.extras.mitcase.grid_family", "using*Grid switch -> grid family", _grid_switch
    ),
    ("PARM04", "usingcylindricalgrid"): _map(
        "case.extras.mitcase.grid_family", "using*Grid switch -> grid family", _grid_switch
    ),
    ("PARM04", "delr"): _map(
        "grid.vertical_layer_thicknesses",
        "delR verbatim (positive down); grid.depth_m = sum(delR)",
        _f_layers,
    ),
    ("PARM04", "delz"): _map(
        "grid.vertical_layer_thicknesses",
        "delZ verbatim (positive down); grid.depth_m = sum(delZ)",
        _f_layers,
    ),
    ("PARM04", "delrc"): _map(
        "grid.vertical_center_spacing (+ vertical_layer_thicknesses, depth_m without delR)",
        "skip when equal to the delR midpoints; otherwise drC verbatim (gmredi2/4)",
        _f_delrc,
    ),
    ("PARM04", "delx"): _map(
        "grid.resolution_deg | case.extras.dx",
        "grid-family derived; uniform spacing only",
        lambda v, ctx: _horizontal_spacing(v, ctx, "x"),
    ),
    ("PARM04", "dely"): _map(
        "grid.resolution_deg | case.extras.dy",
        "grid-family derived; uniform spacing only",
        lambda v, ctx: _horizontal_spacing(v, ctx, "y"),
    ),
    ("PARM04", "dxspacing"): _map(
        "grid.resolution_deg | case.extras.dx",
        "grid-family derived",
        lambda v, ctx: _horizontal_spacing(v, ctx, "x"),
    ),
    ("PARM04", "dyspacing"): _map(
        "grid.resolution_deg | case.extras.dy",
        "grid-family derived",
        lambda v, ctx: _horizontal_spacing(v, ctx, "y"),
    ),
    ("PARM04", "xgorigin"): _map(
        "grid.center_lon", "origin + nx*delX/2 (sphere)", lambda v, ctx: _origin(v, ctx, "x")
    ),
    ("PARM04", "ygorigin"): _map(
        "grid.center_lat", "origin + ny*delY/2 (sphere)", lambda v, ctx: _origin(v, ctx, "y")
    ),
    ("PARM04", "delxfile"): _map(
        "case.extras.mitcase.mesh_spacing.del_x_file",
        "Cartesian exact-one-record delX file at readBinaryPrec; spherical defer",
        lambda v, ctx: _spacing_file(v, ctx, "x"),
    ),
    ("PARM04", "delyfile"): _map(
        "case.extras.mitcase.mesh_spacing.del_y_file",
        "Cartesian exact-one-record delY file at readBinaryPrec; spherical defer",
        lambda v, ctx: _spacing_file(v, ctx, "y"),
    ),
    ("PARM04", "horizgridfile"): _gap("curvilinear horizontal grid file"),
    ("PARM04", "rsphere"): _map(
        "case.extras.r_sphere + flux_form_momentum.recip_rsphere",
        "identity, and recip_rSphere = 1/rSphere (ini_parms.F:1362), on a "
        "spherical-polar grid; no reader on Cartesian or cylindrical; curvilinear gap",
        _f_rsphere,
    ),
    ("PARM04", "radius_fromhorizgrid"): _map(
        "grid",
        "no reader outside a curvilinear grid, which is a gap",
        _f_radius_from_horiz_grid,
    ),
    ("PARM04", "deepatmosphere"): _false_or_gap("deep-atmosphere geometry"),
    ("PARM04", "top_pres"): _gap("pressure-coordinate top pressure"),
    ("PARM04", "pcellmix_select"): _gap("partial-cell mixing (pCellMix)"),
    ("PARM04", "pcellmix_delr"): _gap("partial-cell mixing (pCellMix)"),
    ("PARM04", "pcellmix_diffkr"): _gap("partial-cell mixing (pCellMix)"),
    ("PARM04", "pcellmix_viscar"): _gap("partial-cell mixing (pCellMix)"),
    # ---- PARM05: input datasets ----------------------------------------
    ("PARM05", "bathyfile"): _ic_binary_input("bathy_file", "bathymetry"),
    ("PARM05", "hydrogthetafile"): _ic_binary_input("temp_file", "initial temperature"),
    ("PARM05", "hydrogsaltfile"): _ic_binary_input("salt_file", "initial salinity"),
    ("PARM05", "uvelinitfile"): _ic_binary_input("u_file", "initial u"),
    ("PARM05", "vvelinitfile"): _ic_binary_input("v_file", "initial v"),
    ("PARM05", "psurfinitfile"): _ic_binary_input(
        "eta_file", "initial surface pressure / eta"
    ),
    ("PARM05", "zonalwindfile"): _forcing_binary_input(
        "zonal_wind_file", "zonal wind stress"
    ),
    ("PARM05", "meridwindfile"): _forcing_binary_input(
        "merid_wind_file", "meridional wind stress"
    ),
    ("PARM05", "surfqnetfile"): _forcing_binary_input(
        "surf_qnet_file", "net surface heat flux (Qnet, sign per §3)"
    ),
    ("PARM05", "surfqfile"): _unread_binary_input(
        "surface heat flux",
        "MITgcm reads it into Qnet ahead of surfQnetFile (ini_forcing.F:77-80, "
        "external_fields_load.F:126-134), so surfQnetFile's source would read the other file",
    ),
    ("PARM05", "surfqswfile"): _unread_binary_input(
        "shortwave surface heat flux",
        "MITgcm reads it into Qsw under SHORTWAVE_HEATING (ini_forcing.F:114-117, "
        "external_fields_load.F:174-193) and adds it to a surfQFile Qnet",
    ),
    ("PARM05", "empmrfile"): _forcing_binary_input(
        "empmr_file", "evaporation minus precipitation (EmPmR)"
    ),
    ("PARM05", "thetaclimfile"): _clim_file("theta_clim_file", "surface T climatology"),
    ("PARM05", "saltclimfile"): _clim_file("salt_clim_file", "surface S climatology"),
    ("PARM05", "topofile"): _gap("surface topography input (atmosphere / p-coordinate)"),
    ("PARM05", "geopotanomfile"): _gap("geopotential anomaly input"),
    ("PARM05", "addmassfile"): _gap("mass source input (addFluid)"),
    # ini_forcing.F:191-195 reads pLoad once; external_forcing_surf.F:376-381 sets
    # phi0surf = pLoad*recip_rhoConst, which calc_grad_phi_hyd.F:127-133 adds to
    # the potential before its gradient (start/1: surface_pressure_load).
    ("PARM05", "ploadfile"): _forcing_binary_input(
        "pload_file", "atmospheric pressure load (pLoad, Pa)"
    ),
    ("PARM05", "the_run_name"): _map("case.extras.description"),
    ("PARM05", "checkinitemp"): _stage(
        "check_ini_temp", "MIT INI_THETA freezing-clamp gate"
    ),
    ("PARM05", "tcylin"): _gap("cylindrical-geometry boundary temperature"),
    ("PARM05", "tcylout"): _gap("cylindrical-geometry boundary temperature"),
}


# --------------------------------------------------------------------------
# THE SHARED MAPPING TABLE -- ``input/eedata``
# --------------------------------------------------------------------------

EEDATA_RULES: dict[tuple[str, str], Rule] = {
    ("EEPARMS", "ntx"): _skip("MIT OpenMP tile threading"),
    ("EEPARMS", "nty"): _skip("MIT OpenMP tile threading"),
    ("EEPARMS", "usesetrlstk"): _skip("MIT stack-limit call"),
    ("EEPARMS", "usecubedsphereexchange"): _false_or_gap(
        "cubed-sphere exchange topology (LOCOMgcm cube path is outside this translator)"
    ),
    ("EEPARMS", "debugmode"): _skip("MIT debug verbosity"),
    ("EEPARMS", "usecoupler"): _false_or_gap("MIT coupler component"),
    ("EEPARMS", "printmapincludeszeros"): _skip("MIT print formatting"),
    ("EEPARMS", "maxlengthprt1d"): _skip("MIT print formatting"),
}
