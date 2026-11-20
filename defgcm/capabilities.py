from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCapabilities:
    supports_sigma: bool = False
    supports_open_boundary: bool = False
    supports_split_explicit: bool = False
    supports_fullgraph_model_step: bool = False
    supports_finite_volume_tracers: bool = False
    supports_ml_closure: bool = False
    supports_nesting: bool = False
    supports_hydrostatic_3d: bool = False
    supports_barotropic_2d: bool = True
    supports_baroclinic_pressure_gradient: bool = False
    supports_diagnostic_vertical_velocity: bool = False
    supports_vertical_tracer_transport: bool = False
    supports_vertical_mixing: bool = False
    # True when the stepper has KPP (Large et al. 1994) vertical mixing enabled
    # (ImplicitFreeSurfaceStepper(kpp_enabled=True)): per-step diagnosed
    # diffusivities + ghat nonlocal transport, op-level aligned vs MITgcm.
    supports_kpp: bool = False
    # Arithmetic of KPP's adjacent-layer density difference (dbloc) when KPP is on:
    # "statekpp" (MITgcm STATEKPP, two densities subtracted) or "divided_difference"
    # (JMD95 divided difference); "n/a" without KPP.  From the stepper's
    # kpp_dbloc_form (defgcm.dynamics.kpp.KPP_DBLOC_FORMS).
    kpp_dbloc_form: str = "n/a"
    supports_coriolis: bool = False
    # Stepper-mode flag: True iff the stepper is an ImplicitFreeSurfaceStepper
    # with stagger_tracer_stepping=True. This is a MODE marker derived purely
    # from stepper configuration (see model._build_capabilities); it does NOT
    # verify that the individual MIT-aligned modules (OBC inflow fix,
    # TsurfCor/SsurfCor, full-precision HPG) are actually wired in — those are
    # expected to accompany that stepper mode but are not checked here.
    supports_mitgcm_strict_alignment: bool = False
