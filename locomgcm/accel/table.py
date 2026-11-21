"""Implementation identities and bounded, per-implementation qualifications.

A tier describes the derivative mechanism, not qualification. SOURCE and
NOT_RUN cells are deliberately inadmissible; a reference observation is never
copied to a compiled or native row. Configuration/cache rows name their real
resource owner instead of inventing additional mathematical operators.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal, Mapping

Tier = Literal["A", "B", "C"]
Backend = Literal["torch_reference", "torch_compiled", "native"]
BackwardSource = Literal["framework_ad", "native_kernel", "analytic_adjoint",
                         "implicit_checked_root", "reference_recompute", "none"]
CapabilityStatus = Literal["PASS", "SOURCE", "FAIL", "NOT_RUN"]


@dataclass(frozen=True)
class ImplementationRow:
    operator: str
    selector: str
    tier: Tier
    backend: Backend
    backward_source: BackwardSource
    entry: str
    reference_selector: str | None = None
    forward: CapabilityStatus = "SOURCE"
    vjp: CapabilityStatus = "NOT_RUN"
    jvp: CapabilityStatus = "NOT_RUN"
    vmap: CapabilityStatus = "NOT_RUN"
    compile: CapabilityStatus = "NOT_RUN"
    remat: CapabilityStatus = "NOT_RUN"
    multi_rank: CapabilityStatus = "NOT_RUN"
    evidence: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        # Freeze the evidence with the selected identity; callers cannot upgrade
        # the qualification by mutating a dictionary after admission.
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


# Communication is selected by an explicit session, independently of kernel switches.
COMMUNICATION_IMPLEMENTATIONS = tuple(
    ImplementationRow(
        operator=f"communication.{name}", selector="comm_session", tier="A",
        backend="torch_reference", backward_source="analytic_adjoint",
        entry=f"locomgcm.distributed.comm_adjoint.{name}",
        forward="PASS", vjp="PASS", compile="PASS", remat="PASS", multi_rank="PASS",
        evidence={
            "forward": "halo exchange adjoint check: regional/cube CPU float64 forward and VJP",
            "vjp": "mixed remote-control VJP check: CPU float64, Gloo, 2 ranks",
            "compile": "mixed remote-control VJP check: eager/aot_eager fullgraph, prepared receive inputs",
            "remat": "mixed remote-control VJP check: explicit session.checkpoint",
            "multi_rank": "model step matches the single-domain step: CPU float64, 2 ranks",
        },
    ) for name in ("halo_transfer", "ordered_sum")
)


# CUDA transport has its own evidence; CPU fullgraph qualifications are not
# inherited. Whole-model compilation is measured by the h160 case separately.
CUDA_COMMUNICATION_IMPLEMENTATIONS = tuple(
    replace(row, compile="NOT_RUN", evidence={
        key: "distributed cg3d ROOT adjoint and recompute check: "
             "FP64 NCCL, x/y decomposition, remote objective, mixed ROOT/halo/SUM and remat"
        for key in ("forward", "vjp", "remat", "multi_rank")
    }) for row in COMMUNICATION_IMPLEMENTATIONS
)


C_SELECTORS = frozenset(("tridiag_cuda", "cg3d_fast_thomas", "cg2d_fused_pcg",
                         "cg3d_fused_pcg", "ivadv"))

# Sea-ice selectors are qualified on their standalone layout, FP64, single rank,
# forward only. Off CUDA, auto selects the reference rows; each native row holds
# its own CUDA evidence, and admission's backend check keeps it CUDA-only.
_SEAICE_REFERENCE_EVIDENCE = {
    "forward": "offline_exf_seaice dyn_lsr and base programs, CUDA eager and CPU model_step: "
               "PASS against MITgcm, float64",
    "compile": "offline_exf_seaice dyn_lsr and base programs, CPU model_step: PASS against MITgcm, float64",
}
_SEAICE_NATIVE_EVIDENCE = {
    "forward": "offline_exf_seaice dyn_lsr 3 steps and base 2 steps, CUDA float64 eager: outputs with both "
               "kernels equal the reference byte for byte",
    "compile": "offline_exf_seaice model_step, CUDA float64: kernels on equal kernels off byte for byte "
               "(dyn_lsr 3, base 4 steps); base program with kernels on PASS against MITgcm",
}
_SEAICE_GRAPH_EVIDENCE = {
    "forward": "offline_exf_seaice dyn_lsr 3 steps and base 2 steps, CUDA float64 eager: replayed blocks "
               "equal the per-call loop byte for byte, iteration records equal",
    "compile": "offline_exf_seaice dyn_lsr 3 steps and base 2 steps, CUDA float64 model_step: replayed "
               "blocks equal the per-call loop byte for byte, iteration records equal",
}

# Besides the sea-ice rows, only these two CPU reference legs are qualified. The
# evidence names the delivered executable qualification and its layout/control
# limits. All other rows retain SOURCE/NOT_RUN, including every other
# native/compiled capability.
_REFERENCE_EVIDENCE = {
    "impldiff_build_batch": {
        "forward": "reference qualification: diffusion; CPU float64, latlon, 1x3x2x2, batched vs loop exact, initial RHS",
        "vjp": "reference qualification: diffusion; CPU float64, initial RHS analytic cotangent, fixed diffusivity/geometry",
    },
    "tracer_adv_fused_cuda": {
        "forward": "reference qualification: tracer; CPU float64, periodic latlon, DST3 X/Y, eager reference",
        "vjp": "reference qualification: tracer; CPU float64, periodic latlon, initial tracer/u/v directional finite differences",
    },
    "seaice_lsr_cuda": _SEAICE_REFERENCE_EVIDENCE,
    "seaice_ordered_sum_cuda": _SEAICE_REFERENCE_EVIDENCE,
    "seaice_lsr_graph": _SEAICE_REFERENCE_EVIDENCE,
}

# This is a static qualification domain, not a property inferred from tensors
# encountered by a dispatcher. Native eligibility still requires its own PASS.
QUALIFIED_DOMAINS = MappingProxyType({
    **{key: (("cpu",), "torch.float64", "latlon", frozenset(("initial",)))
       for key in ("impldiff_build_batch", "tracer_adv_fused_cuda")},
    **{key: (("cpu", "cuda"), "torch.float64", "seaice_native", frozenset())
       for key in ("seaice_lsr_cuda", "seaice_ordered_sum_cuda", "seaice_lsr_graph")},
})


def _reference(operator: str, selector: str, entry: str, *, tier: Tier = "A") -> ImplementationRow:
    evidence = _REFERENCE_EVIDENCE.get(selector, {})
    return ImplementationRow(operator=operator, selector=selector, tier=tier,
        backend="torch_reference",
        backward_source="analytic_adjoint" if tier == "C" else "framework_ad",
        entry=entry, forward="PASS" if "forward" in evidence else "SOURCE",
        vjp="PASS" if "vjp" in evidence else "NOT_RUN",
        compile="PASS" if "compile" in evidence else "NOT_RUN", evidence=evidence)


def _accelerated(operator: str, selector: str, entry: str, *, tier: Tier,
                 backend: Backend = "native", backward_source: BackwardSource = "none",
                 evidence: Mapping[str, str] = MappingProxyType({})) -> ImplementationRow:
    return ImplementationRow(operator=operator, selector=selector, tier=tier,
        backend=backend, backward_source=backward_source, entry=entry,
        reference_selector=selector, forward="PASS" if "forward" in evidence else "SOURCE",
        compile="PASS" if "compile" in evidence else "NOT_RUN", evidence=evidence)


IMPLEMENTATION_TABLE: Mapping[str, tuple[ImplementationRow, ...]] = MappingProxyType({
    'cache_constants': (
        _reference('cache.constants', 'cache_constants',
                   'locomgcm.accel.constant_cache.ConstantCache.get_or_build'),
    ),
    'cg2d_cuda_graph': (
        _reference('pressure.cg2d', 'cg2d_cuda_graph',
                   'locomgcm.solvers.cg2d.pcg_solve'),
        _accelerated('pressure.cg2d', 'cg2d_cuda_graph',
                     'locomgcm.solvers.cg2d.pcg_solve_graphed', tier='B', backend="torch_compiled"),
    ),
    'cg2d_fused_pcg': (
        _reference('pressure.cg2d', 'cg2d_fused_pcg',
                   'locomgcm.solvers.cg2d.pcg_solve'),
        _accelerated('pressure.cg2d', 'cg2d_fused_pcg',
                     'locomgcm.solvers.cg2d.pcg_solve_fused', tier='C'),
    ),
    'cg3d_fused_pcg': (
        _reference('pressure.cg3d', 'cg3d_fused_pcg',
                   'locomgcm.solvers.cg3d.CG3DSolver.solve'),
        _accelerated('pressure.cg3d', 'cg3d_fused_pcg',
                     'locomgcm.solvers.cg3d.CG3DSolver._fused_iteration', tier='C'),
        _accelerated('pressure.cg3d', 'cg3d_fused_pcg',
                     'locomgcm.solvers._cg3d_fused.graph.cg3d_graphed_iteration', tier='C', backend="torch_compiled"),
    ),
    'cg3d_fast_thomas': (
        _reference('pressure.cg3d_preconditioner', 'cg3d_fast_thomas',
                   'locomgcm.solvers.cg3d_preconditioner.apply_cg3d_preconditioner'),
        _accelerated('pressure.cg3d_preconditioner', 'cg3d_fast_thomas',
                     'locomgcm.solvers._cg3d_thomas._fast.apply_cg3d_preconditioner_fast', tier='C', backward_source="analytic_adjoint"),
    ),
    'compile_dynamic': (
        replace(_reference('configuration.compile_dynamic', 'compile_dynamic',
                           'experiments.cases._runtime._resolve_compile_dynamic'),
                vjp="FAIL", jvp="FAIL", evidence={
                    axis: "refused by locomgcm.runner.refuse_dynamic_derivative at the "
                          "runner and checkpoint-window entries: a forward-only diagnostic; "
                          "a differentiated dynamic-shape f32 scs_paper_h160 step compiled "
                          "154 graphs over hours and then failed inside Inductor"
                    for axis in ("vjp", "jvp")}),
    ),
    'compile_pin_schedule': (
        _reference('configuration.compile_pin_schedule', 'compile_pin_schedule',
                   'experiments.cases._runtime._resolve_compile_pin_schedule'),
    ),
    'compile_fullgraph': (
        _reference('configuration.compile_fullgraph', 'compile_fullgraph',
                   'locomgcm.runner._configure_step_execution'),
    ),
    'cube_local_grid_cache': (
        _reference('cache.cube_local_grid', 'cube_local_grid_cache',
                   'locomgcm.geometry.cube_localize.localize_cube_grid'),
    ),
    'device_resident_masks': (
        _reference('cache.device_geometry', 'device_resident_masks',
                   'locomgcm.geometry.grid.ensure_grid_device_resident'),
    ),
    'impldiff_build_batch': (
        _reference('vertical_diffusion.assembly', 'impldiff_build_batch',
                   'locomgcm.solvers.tridiag.build_implicit_diffusion_system'),
    ),
    'ivadv': (
        _reference('vertical_advection.band_solve', 'ivadv',
                   'locomgcm.solvers.implicit_vertical.solve_tridiagonal_system'),
        _accelerated('vertical_advection.band_solve', 'ivadv',
                     'locomgcm.solvers.implicit_vertical_fused.solve_tracer_band_thomas_fused', tier='C', backward_source="native_kernel"),
    ),
    'kpp_face_batch': (
        _reference('kpp.face_execution', 'kpp_face_batch',
                   'locomgcm.dynamics.kpp_mixing.cube_kpp_fields'),
    ),
    'kpp_static_halo_cache': (
        _reference('cache.kpp_static_halo', 'kpp_static_halo_cache',
                   'locomgcm.dynamics.kpp_mixing._static_halo'),
    ),
    'kpp_wscale_batch': (
        _reference('kpp.velocity_scale', 'kpp_wscale_batch',
                   'locomgcm.dynamics.kpp.kpp_bldepth'),
    ),
    'vectorize_vertshear': (
        _reference('momentum.vertical_shear', 'vectorize_vertshear',
                   'locomgcm.dynamics.vector_invariant_momentum.vertical_shear_tendency_for_grid'),
    ),
    'tridiag_cuda': (
        _reference('tridiagonal.solve', 'tridiag_cuda',
                   'locomgcm.solvers.tridiag.tridiag_solve', tier="C"),
        _accelerated('tridiagonal.solve', 'tridiag_cuda',
                     'locomgcm.solvers._tridiag_cuda.dispatcher.tridiag_solve', tier='C'),
    ),
    'tridiag_build_cuda': (
        _reference('vertical_diffusion.assembly', 'tridiag_build_cuda',
                   'locomgcm.solvers.tridiag.build_implicit_diffusion_system'),
        _accelerated('vertical_diffusion.assembly', 'tridiag_build_cuda',
                     'locomgcm.solvers._tridiag_build_cuda.dispatcher.build_implicit_diffusion_system', tier='B'),
    ),
    'kpp_bldepth_cuda': (
        _reference('kpp.boundary_layer_depth', 'kpp_bldepth_cuda',
                   'locomgcm.dynamics.kpp.kpp_bldepth'),
        _accelerated('kpp.boundary_layer_depth', 'kpp_bldepth_cuda',
                     'locomgcm.dynamics._kpp_bldepth_cuda.dispatcher.kpp_bldepth_cuda',
                     tier='A', backward_source='native_kernel'),
    ),
    'kpp_blmix_cuda': (
        # Native profile VJP; the existing wscale preparation retains Torch AD.
        _reference('kpp.boundary_layer_mixing', 'kpp_blmix_cuda',
                   'locomgcm.dynamics.kpp.kpp_blmix'),
        _accelerated('kpp.boundary_layer_mixing', 'kpp_blmix_cuda',
                     'locomgcm.dynamics._kpp_blmix_cuda.dispatcher.kpp_blmix',
                     tier='A', backward_source='native_kernel'),
    ),
    'kpp_statekpp_cuda': (
        _reference('kpp.density', 'kpp_statekpp_cuda',
                   'locomgcm.dynamics.kpp.kpp_statekpp'),
        _accelerated('kpp.density', 'kpp_statekpp_cuda',
                     'locomgcm.dynamics._kpp_statekpp_cuda.dispatcher.kpp_statekpp_cuda',
                     tier='A', backward_source='native_kernel'),
    ),
    'kpp_fused_cuda': (
        _reference('kpp.mixing', 'kpp_fused_cuda',
                   'locomgcm.dynamics.kpp.kpp_calc_diffusivities'),
        _accelerated('kpp.mixing', 'kpp_fused_cuda',
                     'locomgcm.dynamics._kpp_fused.dispatcher.regional_kpp_fused_dispatch', tier='B'),
    ),
    'dst3_cuda': (
        _reference('tracer.dst3_flux', 'dst3_cuda',
                   'locomgcm.operators.fluxes.tracer_flux.horizontal_stencil_flux'),
        _accelerated('tracer.dst3_flux', 'dst3_cuda',
                     'locomgcm.operators.fluxes._dst3_cuda.dispatcher.dst3_transport_flux_values_cuda',
                     tier='A', backward_source='native_kernel'),
    ),
    'eos_cuda': (
        _reference('equation_of_state.density', 'eos_cuda',
                   'locomgcm.dynamics._eos_cuda.dispatcher._eager'),
        _accelerated('equation_of_state.density', 'eos_cuda',
                     'locomgcm.dynamics._eos_cuda.dispatcher.density_fields',
                     tier='A', backward_source='native_kernel'),
    ),
    'vim_coriolis_u': (
        _reference('momentum.coriolis_u', 'vim_coriolis_u',
                   'locomgcm.dynamics._vim_coriolis_u_cuda.dispatcher._reference'),
        _accelerated('momentum.coriolis_u', 'vim_coriolis_u',
                     'locomgcm.dynamics._vim_coriolis_u_cuda.dispatcher.u_face_coriolis_from_haloed_v_row_metrics_cuda', tier='B'),
    ),
    'vim_fused_cuda': (
        _reference('momentum.vector_invariant', 'vim_fused_cuda',
                   'locomgcm.dynamics.vector_invariant_momentum.VectorInvariantMomentumAdvection.evaluate'),
        _accelerated('momentum.vector_invariant', 'vim_fused_cuda',
                     'locomgcm.dynamics._vim_fused.dispatcher.vim_fused_dispatch', tier='B'),
    ),
    'tracer_adv_fused_cuda': (
        _reference('tracer.horizontal_advection', 'tracer_adv_fused_cuda',
                   'locomgcm.dynamics.tracers.HorizontalTracerAdvection._multidim_advective_tendency'),
        _accelerated('tracer.horizontal_advection', 'tracer_adv_fused_cuda',
                     'locomgcm.dynamics._tracer_adv_fused.dispatcher._run_horizontal', tier='B'),
    ),
    'hvisc_fused_cuda': (
        _reference('momentum.horizontal_viscosity', 'hvisc_fused_cuda',
                   'locomgcm.dynamics.horizontal_viscosity.HorizontalViscosity.evaluate'),
        _accelerated('momentum.horizontal_viscosity', 'hvisc_fused_cuda',
                     'locomgcm.dynamics._hvisc_fused.dispatcher.hvisc_fused_dispatch', tier='B'),
    ),
    'seaice_lsr_cuda': (
        _reference('seaice.lsr_line_recurrence', 'seaice_lsr_cuda',
                   'locomgcm.seaice.dynamics.sweep.reference_sweep'),
        _accelerated('seaice.lsr_line_recurrence', 'seaice_lsr_cuda',
                     'locomgcm.seaice.dynamics._lsr_cuda.dispatcher.sweep_both', tier='B',
                     evidence=_SEAICE_NATIVE_EVIDENCE),
    ),
    'seaice_ordered_sum_cuda': (
        _reference('seaice.ordered_residual_sum', 'seaice_ordered_sum_cuda',
                   'locomgcm.seaice.space._ordered_sum'),
        _accelerated('seaice.ordered_residual_sum', 'seaice_ordered_sum_cuda',
                     'locomgcm.seaice.dynamics._lsr_cuda.dispatcher.ordered_sum', tier='B',
                     evidence=_SEAICE_NATIVE_EVIDENCE),
    ),
    'seaice_lsr_graph': (
        _reference('seaice.lsr_linear_iterations', 'seaice_lsr_graph',
                   'locomgcm.seaice.dynamics.lsr.solve_lsr'),
        _accelerated('seaice.lsr_linear_iterations', 'seaice_lsr_graph',
                     'locomgcm.seaice.dynamics._lsr_cuda.graph.LinearBlocks.run', tier='B',
                     evidence=_SEAICE_GRAPH_EVIDENCE),
    ),
})


@dataclass(frozen=True)
class CacheEntry:
    selector: str
    site: str
    payload: str
    admission: Literal["static_only", "reject_controls", "recompute_on_control"]


# Thirteen shared-cache call sites, plus the cube/device resource caches.
# Control-bearing entries are rejected; recomputation may be qualified without
# converting graph-bearing values into static cache entries.
CACHE_ENTRIES: Mapping[str, tuple[CacheEntry, ...]] = MappingProxyType({
    "cache_constants": (
        CacheEntry("cache_constants", "locomgcm/geometry/masks.py:283", "fixed center/u/v masks", "static_only"),
        CacheEntry("cache_constants", "locomgcm/dynamics/coriolis.py:171", "fixed latitude Coriolis profile", "reject_controls"),
        CacheEntry("cache_constants", "locomgcm/dynamics/spherical_metric.py:80", "fixed latitude tan_phi_u", "static_only"),
        CacheEntry("cache_constants", "locomgcm/dynamics/spherical_metric.py:85", "fixed latitude tan_phi_v", "static_only"),
        CacheEntry("cache_constants", "locomgcm/dynamics/_vector_invariant_topology.py:219", "fixed corner Coriolis profile", "reject_controls"),
        CacheEntry("cache_constants", "locomgcm/dynamics/_vector_invariant_topology.py:188", "fixed metric halos and decomposition", "static_only"),
        CacheEntry("cache_constants", "locomgcm/dynamics/vector_invariant_momentum.py:602", "static hFac-derived corner fractions; dynamic hFac bypasses", "static_only"),
        CacheEntry("cache_constants", "locomgcm/dynamics/viscosity_geometry.py:673", "static reciprocal hFac and masks; dynamic hFac bypasses", "static_only"),
        CacheEntry("cache_constants", "locomgcm/dynamics/viscosity_geometry.py:759", "static corner hFacZ; dynamic hFac bypasses", "static_only"),
        CacheEntry("cache_constants", "locomgcm/dynamics/viscosity_geometry.py:819", "fixed Leith metric length scales, excluding physical coefficients", "static_only"),
        CacheEntry("cache_constants", "locomgcm/dynamics/momentum_sponge.py:429", "fixed boundary indices/side masks, excluding targets and timescales", "static_only"),
        CacheEntry("cache_constants", "locomgcm/dynamics/kpp_mixing.py:195", "background diffusivity: Python float Ks/Kt memoized; explicit tensor Ks/Kt recomputed", "recompute_on_control"),
    ),
    "kpp_static_halo_cache": (
        CacheEntry("kpp_static_halo_cache", "locomgcm/dynamics/kpp_mixing.py:156", "fixed-grid maskC/Coriolis/kmtj halos; explicit Coriolis recomputed", "recompute_on_control"),
    ),
    "cube_local_grid_cache": (
        CacheEntry("cube_local_grid_cache", "locomgcm/geometry/cube_localize.py:115", "fixed local geometry and rank decomposition", "static_only"),
    ),
    "device_resident_masks": (
        CacheEntry("device_resident_masks", "locomgcm/geometry/grid.py:134", "fixed device geometry and masks", "static_only"),
    ),
})
