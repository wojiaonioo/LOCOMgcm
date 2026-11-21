"""Registry of accelerator capabilities and dispatch policies."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from .schema import AcceleratorSpec as _AcceleratorSpec


@dataclass(frozen=True)
class AcceleratorSpec(_AcceleratorSpec):
    """Accelerator declaration with an optional deferred extension loader."""

    preload: str | None = None


_CLIENTS = (
    AcceleratorSpec(
        key="cache_constants",
        env_var="LOCOMGCM_CACHE_CONSTANTS",
        default=True,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon", "cube"),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
    ),
    AcceleratorSpec(
        key="cg2d_cuda_graph",
        env_var="LOCOMGCM_CG2D_CUDA_GRAPH",
        default=False,
        # Admission already routes unsupported solves to the eager/ROOT
        # reference.  Decline records must preserve that policy; selected
        # capture/replay failures are still recorded as errors and re-raised.
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon", "cube"),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
    ),
    AcceleratorSpec(
        # The CG2D PCG iteration body as fused kernels: the five-point
        # Helmholtz apply, the x/r update (with the Jacobi preconditioner
        # folded in), the SSOR preconditioner apply, and the search-direction
        # update.  Consumer boundary = ``cg2d.pcg_solve_fused``, reached from
        # ``implicit_free_surface_cg2d.cg2d_solve``.
        #
        # SUPERSEDES ``cg2d_cuda_graph`` in intent: the hand-recorded CUDA
        # graph accelerates LAUNCHING the unfused body, this replaces the body.
        # They are not composed -- the call site tries this one first and only
        # falls through to the graph when it declines; the graph remains
        # available while both execution paths are supported.
        #
        # ``grid_families=("latlon", "cube")`` and
        # ``distributed=("single", "multi")`` : the solver
        # layer imposes NO restriction of either kind.  The gate tests only what
        # the kernels can index -- device, dtype, contiguity, rank, extent,
        # grad -- so a cube problem is declined by the shape of its
        # face-resolved coefficients and an N-rank problem is admitted for the
        # same reason a 1-rank problem is: nothing in the predicate can see the
        # difference.  Reachability is the INJECTION layer's obligation (the
        # halo, the reducer and the coefficient assembly the caller supplies)
        # and is tracked there, not by narrowing this declaration.  The
        # question asked of a family is "does the solver restrict it", not
        # "is it reachable today".
        #
        # capability + fallback, not correctness + raise: declining is always
        # safe here (the eager ``pcg_solve`` is the bit-identical reference),
        # and a ``raise`` policy would abort production runs, as it does for
        # the two correctness-class CG keys.
        #
        # Path precedence:
        # When ``cg2d_cuda_graph`` admits the solve the stepper
        # takes the graph replay and records this key ``inapplicable`` with
        # ``reason=superseded_by_cg2d_cuda_graph`` -- same tree, same switch
        # set, 2xA800 N=960: fused body 472.9 ms/step vs replay 336.5.  The
        # fused loop keeps two host reads per iteration; until it carries the
        # graph's block control flow  it is the accelerated
        # path only where the graph declines (grad, non-CUDA, other topologies).
        # A ROOT call (the differentiated solve, an atomic round inside a
        # CommSession) never takes the fused body: the stepper records
        # ``reason=superseded_by_root_adjoint`` and runs the ROOT path.
        key="cg2d_fused_pcg",
        env_var="LOCOMGCM_CG2D_FUSED_PCG",
        default=False,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon", "cube"),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
        preload="locomgcm.solvers._cg2d_fused:load_extension",
    ),
    AcceleratorSpec(
        # The CG3D PCG iteration body as fused kernels on the shared
        # topology-free core (``solvers/_pcg_core``): the seven-point apply from
        # a padded block, the search-direction update, the x/r update, and the
        # vertical-line Thomas apply with the mask folded onto its store.
        # In no-grad production, the same kernels and halo collectives replay
        # one captured iteration; one packed readback preserves all stop tests.
        # Consumer boundary = ``cg3d.CG3DSolver.solve``.
        #
        # ``grid_families=("latlon", "cube")`` / ``distributed=("single",
        # "multi")`` : the solver layer restricts neither.
        # The kernels take a padded interior block and coefficient tensors and
        # index a plain leading extent, so they cannot tell a face-leading field
        # from a regional one, nor one rank from N.  Reachability on a cube
        # depends on ``build_cg3d_operator``, which raises on a cubed-sphere
        # grid (``cg3d_operator.py:116``) -- an INJECTION-layer obligation
        # tracked there, not a reason to narrow this declaration.
        #
        # Padded-block kernels consume exchanged neighbours independently of
        # grid identity. Declines fall back to the bit-identical eager loop.
        key="cg3d_fused_pcg",
        env_var="LOCOMGCM_CG3D_FUSED_PCG",
        default=False,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon", "cube"),
        distributed=("single", "multi"),
        hydrostatic="nonhydrostatic_only",
        gate_satisfiable=True,
        production_reachable=True,
        preload="locomgcm.solvers._cg3d_fused:load_extension",
    ),
    AcceleratorSpec(
        key="cg3d_fast_thomas",
        env_var="LOCOMGCM_CG3D_USE_FAST_THOMAS",
        default=True,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon", "cube"),
        distributed=("single", "multi"),
        hydrostatic="nonhydrostatic_only",
        gate_satisfiable=True,
        production_reachable=True,
    ),
    AcceleratorSpec(
        # Not an accelerator kernel and not a speed lever: a *diagnostic*
        # switch over the runner's compile shape posture.  Static shapes are
        # the production posture,
        # so the default is False and the switch is no
        # longer part of the production posture; setting it to 1 asks for an
        # explicitly dynamic trace, which is how a symbolic-shape defect is
        # reproduced.  The recompile-limit headroom that once motivated
        # dynamic shapes now comes from the posture-independent
        # RECOMPILE_LIMIT in locomgcm/runner.py, without Inductor's
        # unbound-symbol defect.  Registered for the unified env grammar and
        # the requested-hit ignition ledger; keep it out of any "accelerator
        # kernel" head-count.
        #
        # production_reachable stays True: leaving the production posture is
        # not the same as being unreachable there.  A forward-only production
        # run that exports LOCOMGCM_COMPILE_DYNAMIC=1 with a compile target other
        # than "disabled" does hand the posture to torch.compile and does record
        # a hit (locomgcm/runner.py _hold_compile_config), so a requested-zero on
        # such a run would still mean a defect.
        #
        # distributed=("single",): on more than one rank the wiring refuses this
        # switch outright, before the first step.  A fully symbolic trace hands
        # locomgcm::halo_p2p symbolic peer and length arguments that its
        # List[int] schema rejects, so rank 0 fails to compile while the others
        # sit in a collective until the watchdog fires.

        # The single-rank restriction is recorded in both metadata and wiring.
        #
        # Forward only: a run whose stated derivative intent is vjp or jvp is
        # refused as well (locomgcm/runner.py refuse_dynamic_derivative, called
        # at ModelRunner.run's entry, the checkpoint window's entry and the
        # compile context), and records inapplicable with UNSUPPORTED_VJP or
        # UNSUPPORTED_JVP.  The hit is recorded where the posture is handed to
        # torch.compile.  A differentiated f32 scs_paper_h160 step compiled
        # with dynamic shapes built 154 graphs over hours and then failed
        # inside Inductor.  This schema has no derivative
        # axis; the refusal lives in the wiring and in the qualification row
        # (accel/table.py, vjp and jvp FAIL).
        key="compile_dynamic",
        env_var="LOCOMGCM_COMPILE_DYNAMIC",
        default=False,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon", "cube"),
        distributed=("single",),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
    ),
    AcceleratorSpec(
        # Preserve rank-consistent launch ordering:
        # LOCOMGCM_COMPILE_PIN_SCHEDULE=1 (the default) switches off Inductor's
        # peak-memory scheduler reordering pass (``reorder_for_peak_memory``)
        # for the compiled step on more than one rank.  The ranks compile
        # different graphs (open boundaries live on one tile), and that pass
        # moved the heavy external kernels to different sides of the halo p2p
        # rounds on each rank, so every such kernel was paid serially by both
        # ranks inside the NCCL wait: 277.8 -> 169.0 ms/step at N=960 on two
        # cards with the pass off (the locality pass is a net win and
        # stays on).  Not a kernel gate: keep it out of every accelerator
        # head-count.
        #
        # distributed=("multi",): on a single rank there is no peer to wait
        # for, so the passes keep torch's defaults there and the switch is
        # inert (no record); the wiring applies it only when world_size > 1.
        key="compile_pin_schedule",
        env_var="LOCOMGCM_COMPILE_PIN_SCHEDULE",
        default=True,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon", "cube"),
        distributed=("multi",),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
    ),
    AcceleratorSpec(
        # LOCOMGCM_COMPILE_FULLGRAPH=1 asks
        # the runner for torch.compile(model.step, fullgraph=True) plus
        # fail_on_recompile_limit_hit, i.e. one Dynamo graph per step with a hard
        # error on any graph break or silent eager fallback.  Not a kernel gate:
        # keep it out of every accelerator head-count.
        key="compile_fullgraph",
        env_var="LOCOMGCM_COMPILE_FULLGRAPH",
        default=False,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon", "cube"),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
    ),
    AcceleratorSpec(
        key="cube_local_grid_cache",
        env_var="LOCOMGCM_CUBE_LOCAL_GRID_CACHE",
        default=True,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("cube",),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
    ),
    AcceleratorSpec(
        key="device_resident_masks",
        env_var="LOCOMGCM_DEVICE_RESIDENT_MASKS",
        default=True,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon", "cube"),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
    ),
    AcceleratorSpec(
        key="impldiff_build_batch",
        env_var="LOCOMGCM_IMPLDIFF_BUILD_BATCH",
        default=True,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon", "cube"),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
    ),
    AcceleratorSpec(
        key="ivadv",
        env_var="LOCOMGCM_IVADV_FUSED_BACKEND",
        default=False,
        gate_class="capability",
        unmet_policy="raise",
        grid_families=("latlon", "cube"),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
        setting_kind="backend",
        preload="locomgcm.solvers._implicit_vertical_cuda._fast:_load_extension",
    ),
    AcceleratorSpec(
        key="kpp_face_batch",
        env_var="LOCOMGCM_KPP_FACE_BATCH",
        default=True,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("cube",),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
    ),
    AcceleratorSpec(
        key="kpp_static_halo_cache",
        env_var="LOCOMGCM_KPP_STATIC_HALO_CACHE",
        default=True,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon", "cube"),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
    ),
    AcceleratorSpec(
        key="kpp_wscale_batch",
        env_var="LOCOMGCM_KPP_WSCALE_BATCH",
        default=True,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon", "cube"),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
    ),
    AcceleratorSpec(
        key="vectorize_vertshear",
        env_var="LOCOMGCM_VECTORIZE_VERTSHEAR",
        default=True,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon", "cube"),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
    ),
    AcceleratorSpec(
        key="tridiag_cuda",
        env_var="LOCOMGCM_TRIDIAG_CUDA",
        default=False,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon", "cube"),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
    ),
    AcceleratorSpec(
        key="tridiag_build_cuda",
        env_var="LOCOMGCM_TRIDIAG_BUILD_CUDA",
        default=False,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon", "cube"),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
    ),
    AcceleratorSpec(
        key="kpp_bldepth_cuda",
        env_var="LOCOMGCM_KPP_BLDEPTH_CUDA",
        default=False,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon",),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
    ),
    AcceleratorSpec(
        key="kpp_blmix_cuda",
        env_var="LOCOMGCM_KPP_BLMIX_CUDA",
        default=False,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon",),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
    ),
    AcceleratorSpec(
        key="kpp_statekpp_cuda",
        env_var="LOCOMGCM_KPP_STATEKPP_CUDA",
        default=False,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon", "cube"),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
    ),
    AcceleratorSpec(
        # Whole regional KPP phase fused behind one custom op: STATEKPP ->
        # preprocessing (smoothed/masked dbloc, Ritop, dVsq, shsq, ustar,
        # surface buoyancy) -> bldepth -> Ri interior mixing -> velocity scales
        # -> blmix -> enhance+combine+cell-centre mask, seven launches total.
        # Consumer boundary = ``regional_kpp_fields``; ``cube_kpp_fields`` is
        # untouched, so "latlon" is the honest family claim.
        #
        # SUPERSEDES ``kpp_statekpp_cuda``, ``kpp_bldepth_cuda`` and
        # ``kpp_blmix_cuda`` in its own scope: a hit here means eager
        # ``kpp_calc_diffusivities`` never runs, so those three never reach
        # their own sites and report requested-zero when co-enabled on this
        # path (they keep their cube / op-level sites).  Mutually exclusive,
        # not additive.  ``kpp_wscale_batch`` is unaffected -- it is a torch
        # gate inside the eager path this one replaces wholesale.
        key="kpp_fused_cuda",
        env_var="LOCOMGCM_KPP_FUSED_CUDA",
        default=False,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon",),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
        preload="locomgcm.dynamics._kpp_fused:load_extension",
    ),
    AcceleratorSpec(
        key="dst3_cuda",
        env_var="LOCOMGCM_DST3_CUDA",
        preload="locomgcm.operators.fluxes._dst3_cuda._fast:compile_dst3_cuda_extension",
        default=False,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon", "cube"),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
    ),
    AcceleratorSpec(
        key="eos_cuda",
        env_var="LOCOMGCM_EOS_CUDA",
        default=False,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon", "cube"),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
        preload="locomgcm.dynamics._eos_cuda._fast:load_eos_cuda_extension",
    ),
    AcceleratorSpec(
        # Vector-invariant momentum, U-face Coriolis / vorticity-flux stencil.
        # The consumer boundary is the REGION row-metrics variant only: both
        # callers (``_vorticity_flux_u``, ``_planetary_only_uface_no_hfac``)
        # take the ``v_ext is not None`` cube branch back out through
        # ``_u_face_coriolis_from_haloed_v`` before reaching it, so "latlon"
        # alone is the honest reachability claim. A second, independent guard
        # (``not grid.metrics.is_uniform_x`` at both call sites) also bypasses
        # this boundary, so reachability is narrower still than "latlon".
        key="vim_coriolis_u",
        env_var="LOCOMGCM_VIM_CORIOLIS_U_CUDA",
        default=False,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon",),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
    ),
    AcceleratorSpec(
        # Vector-invariant momentum, whole horizontal phase fused into one
        # launch (curl -> hFacZ mask -> KE gradients -> planetary Coriolis ->
        # vorticity flux -> combine).  Consumer boundary =
        # ``VectorInvariantMomentumAdvection.evaluate``; the adapter reproduces
        # the region row-metrics path only, so "latlon" is the honest family
        # claim -- both cube leaves are declined by ``config_decline_reason``.
        #
        # SUPERSEDES ``vim_coriolis_u`` in its own scope: this boundary sits
        # UPSTREAM of ``_u_face_vorticity_flux``, so with
        # both gates on this one ignites and ``vim_coriolis_u`` reports
        # requested-zero.  They are not meant to be co-enabled; the pair is
        # mutually exclusive, not additive.
        key="vim_fused_cuda",
        env_var="LOCOMGCM_VIM_FUSED_CUDA",
        default=False,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon",),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
        preload="locomgcm.dynamics._vim_fused:load_extension",
    ),
    AcceleratorSpec(
        # Fused horizontal (X then Y) multidimensional tracer advection.  The
        # consumer boundary is ``_multidim_advective_tendency`` in its
        # ``horizontal_only=True`` form -- the staggered hand-off to the
        # implicit vertical solve -- under a DST3 / DST3-FL scheme.
        #
        # SUPERSEDES ``dst3_cuda`` in its own scope: it absorbs the whole span
        # that gate serves on this path, so with both on this one ignites and
        # ``dst3_cuda`` reports requested-zero for the region X/Y adapters
        # (keeping only its cube sites).  Mutually exclusive, not additive.
        #
        # Region single-patch orchestration only; the cube tile adapter reaches
        # a different flux path, hence "latlon" alone.
        key="tracer_adv_fused_cuda",
        env_var="LOCOMGCM_TRACER_ADV_FUSED_CUDA",
        default=False,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon",),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
        preload="locomgcm.dynamics._tracer_adv_fused:load_extension",
    ),
    AcceleratorSpec(
        # Horizontal viscosity, the Leith/Smag dissipation phase fused into two
        # launches: coefficient assembly (hDiv, corner curl, side mask, tension,
        # Leith gradients, both point measures, both Smag magnitudes, the
        # harmonic assembly plus its base) and the div-vort dissipation stencil.
        # Consumer boundary = ``HorizontalViscosity.evaluate``.
        #
        # No SUPERSEDES relation: no other registry key serves any part of this
        # scope.  The client USES ``cache_constants`` and
        # ``device_resident_masks`` -- it fetches the same three memoized
        # geometry constants the eager path fetches -- which is cooperation,
        # not absorption, and neither of those gates changes state because of
        # this one.
        #
        # Multi-rank uses P/Q kernels split at the eager measure exchanges; all
        # communication remains adapter-side with the original halo contracts.
        key="hvisc_fused_cuda",
        env_var="LOCOMGCM_HVISC_FUSED_CUDA",
        default=False,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("latlon",),
        distributed=("single", "multi"),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
        preload="locomgcm.dynamics._hvisc_fused:load_extension",
    ),
    AcceleratorSpec(
        # The sea-ice LSR line recurrence (SEAICE_LSR_TRIDIAGU/V): the per-pass
        # line factors (bet, CUU) and the ordered Gauss-Seidel sweep of U and V,
        # one CUDA thread per tile and direction, in the reference's operation
        # order with separate roundings (locomgcm/seaice/dynamics/_lsr_cuda).
        # Consumer boundary = the locomgcm_seaice::lsr_factor and ::lsr_sweep
        # operators of locomgcm/seaice/dynamics/sweep.py (variants factor, sweep).
        # grid_families=("seaice_native",): the kernel indexes the four native
        # MIT tiles of locomgcm/seaice/space.py; distributed=("single",): one
        # process owns all tiles. capability + fallback: only an auto request
        # falls back at admission, to the qualified reference (table.py), which
        # computes the identical sequence; an explicit request that cannot be
        # admitted is rejected, and selected failures are re-raised.
        key="seaice_lsr_cuda",
        env_var="LOCOMGCM_SEAICE_LSR_CUDA",
        default=False,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("seaice_native",),
        distributed=("single",),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
        preload="locomgcm.seaice.dynamics._lsr_cuda._fast:load_extension",
    ),
    AcceleratorSpec(
        # The ordered residual sum of SEAICE_RESIDUAL (one accumulator in
        # tile, j, i order), a single-thread fold on the device. Consumer
        # boundary = locomgcm_seaice::ordered_sum of locomgcm/seaice/space.py.
        key="seaice_ordered_sum_cuda",
        env_var="LOCOMGCM_SEAICE_ORDERED_SUM_CUDA",
        default=False,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("seaice_native",),
        distributed=("single",),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
        preload="locomgcm.seaice.dynamics._lsr_cuda._fast:load_extension",
    ),
    AcceleratorSpec(
        # Replayed blocks of LSR linear iterations (seaice_lsr.F:875-987):
        # one CUDA graph per block of sweeps, checks, record rows and X6, with
        # the stop state on the device and one host read per block
        # (locomgcm/seaice/dynamics/_lsr_cuda/graph.py). Consumer boundary = the
        # linear loop of locomgcm/seaice/dynamics/lsr.py:solve_lsr. Same
        # admission as the two kernels; the per-call loop is the reference.
        key="seaice_lsr_graph",
        env_var="LOCOMGCM_SEAICE_LSR_GRAPH",
        default=False,
        gate_class="capability",
        unmet_policy="fallback",
        grid_families=("seaice_native",),
        distributed=("single",),
        hydrostatic="any",
        gate_satisfiable=True,
        production_reachable=True,
        preload="locomgcm.seaice.dynamics._lsr_cuda._fast:load_extension",
    ),
)

REGISTRY = MappingProxyType({spec.key: spec for spec in _CLIENTS})

if len(REGISTRY) != len(_CLIENTS):
    raise RuntimeError("duplicate accelerator key in registry")
