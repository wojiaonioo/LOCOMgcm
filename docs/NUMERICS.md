# Numerical conventions

LOCOMgcm arrays use `[batch, k, j, i]` or `[k, j, i]`, with zero-based indices.
MITgcm uses one-based `(i, j, k)`, so `(i, j, k)` maps to `[k-1, j-1, i-1]`.
The surface is the first vertical level. On the Arakawa C grid, U is on the west
face, V on the south face, tracers and elevation at cell centers, and vorticity at
the southwest corner.

## Signs and coordinates

| Quantity | LOCOMgcm | MITgcm | Interface conversion |
|---|---|---|---|
| Vertical velocity | Downward positive | Upward positive | Negate at the convention boundary |
| Wet ocean depth | Positive | Runtime `Depth.data` positive | None |
| Input bathymetry | Positive depth | `bathyFile` negative in ocean | Use `project_bathymetry_to_mitgcm` |
| Layer-center depth | Positive downward | `rC` negative below surface | Negate |
| Layer-interface depth | Positive downward | `rF` negative below surface | Negate |
| Layer thickness | Positive | `drF` positive | None |
| Net heat input | `heat_flux > 0` heats ocean | `Qnet > 0` cools ocean | `heat_flux = -Qnet` |
| Salt input | `salt_flux > 0` adds salt | `EmPmR > 0` adds salt; `saltFlux > 0` removes salt | Preserve `EmPmR`; negate `saltFlux` |

MITgcm's vertical coordinate factor is `rkSign=-1`; its lower-bound coordinate is
negative depth. A translated vertical kernel must account for these conventions.
Document input conventions at producer-consumer boundaries.

## Required semantics

1. `_cached_w_interface` and `_w_interface_prev` store MITgcm's upward-positive
   convention. Negate the downward-positive diagnostic before writing these caches.
2. Translated vertical-shear and vertical-advection kernels that use MITgcm's
   convention receive upward-positive vertical velocity.
3. LOCOMgcm tracer vertical scans and multidimensional advection use its
   downward-positive convention and may consume the diagnostic directly.
4. `vertical_velocity_from_transport_divergence(bottom_up=True)` returns
   downward-positive velocity; its integration direction does not change the sign
   convention of the result.
5. State vertical velocity and depth conventions in new module docstrings and
   beside new cached quantities.
6. Equal magnitudes with opposite signs across implementations require a
   convention check before changing the discrete algorithm.
7. Bathymetry ingestion declares its sign. Dry points are zeroed before
   `obcs_fix_topo_and_clamp`, or handled by its final nonnegative clamp.
8. New translated vertical kernels explicitly handle `rkSign=-1`; see
   `_vertical_shear_rksign_deltas` for an existing implementation.
9. Convert model bathymetry to MIT input bathymetry only through the designated
   exporter. Do not negate MITgcm's positive runtime depth.
10. Center-to-face reconstruction declares `face_mask` explicitly. `None` declares
    no mask; a tensor applies the outer face mask, not a reconstruction weight.
    Clear invalid dry inputs with selection semantics (`torch.where`), since
    multiplying zero by NaN still yields NaN. Preserve nested mask selections.
11. Horizontal neighbor coupling follows
    `grid.edge_face_couples(axis) OR distributed-neighbor`. Single-device periodic
    seams must retain periodic coupling. Staggered duplicate outer faces are
    skipped when forming periodic and distributed halo transfers. The same
    topology rules apply to grid constants such as Coriolis parameters. Do not
    substitute wall or replicated-edge behavior for the declared topology.
12. Missing required inputs must not silently remove a physical term. Make the
    input required or reject an absent value. Every configured term has an actual
    consumer in the selected scheme; assembly rejects missing, duplicate or
    unevaluated contributions. The existing stepper uses `evaluated_domains` and
    `superseded_module_names`; a future explicit time program must enforce the same
    responsibility per stage and output. Preserve control dependence even when a
    continuous parameter currently has value zero. Each translated configuration
    field must have an identifiable reader. Reject an unsupported upstream variant
    explicitly as UNSUPPORTED rather than deferred, and do not produce a case
    result for it. The nonhydrostatic pressure solve omits PRE_CG3D's
    `EmPmR*mass2rUnit` term (`pre_cg3d.F:88-107`), so a step with
    `use_real_freshwater_flux` whose `wpflux_surface_forcing` EmPmR is nonzero on a
    wet cell, requires grad or carries a forward-mode tangent is refused in the
    stepper's prepare phase (`ifs/nh.py` `refuse_real_freshwater_pressure_term`).
13. A zero open-boundary sponge target is a valid value, not a land marker. Use the
    declared mask. Reference comparisons must identify the upstream sponge
    implementation and its treatment of zero targets and southern time scales.
14. Fortran `SIGN(a, -0.0)` depends on compiler behavior and flags. Set
    `kpp_sign_zero` to the actual reference convention (`signbit` or `positive`)
    after checking the build and a runtime probe. KPP's JMD95P pressure uses the
    same dynamical-pressure convention as the stepper; its Ekman depth uses
    cell-center Coriolis, not the mean of adjacent meridional faces.

Preserve dtype promotion points, term structure and the discrete algorithm.
Summation order, fused multiply-add and reciprocal multiplication are assessed
under [validation](VALIDATION.md), with the relevant conditioning and scope.

## KPP buoyancy differences

STATEKPP's local buoyancy jump `dbloc` takes the in-situ densities of levels
k and k+1 at the pressure of level k+1. The stepper field `kpp_dbloc_form`
selects its arithmetic; it is passed explicitly to `kpp_statekpp` and
`kpp_calc_diffusivities`, which have no default for it.

| `kpp_dbloc_form` | Arithmetic | Use |
|---|---|---|
| `statekpp` (default) | two densities subtracted, as MITgcm STATEKPP (`pkg/kpp/kpp_routines.F:1930`) | comparison with MITgcm, including every MITgcm reference result |
| `divided_difference` | JMD95 polynomial divided difference (`jmd95_density_difference`); JMD95 only | near-neutral derivatives |

The two forms are algebraically equal and differ by a few ULP of `g*rho/rho`.
Where `|dbloc|` is about 1e-13 m s-2, below the boundary layer in near-neutral
water, that is 1e-3 relative, and with weak shear the interior Richardson
mixing turns it into viscosity differences of order 1e-5 m2 s-1. The
subtraction form can lose derivative accuracy in this near-neutral regime;
the divided difference avoids the density subtraction. The CUDA STATEKPP kernel (`LOCOMGCM_KPP_STATEKPP_CUDA`) and the fused KPP
kernel (`LOCOMGCM_KPP_FUSED_CUDA`) compute the divided difference for JMD95 and
decline the default form as `inapplicable:dbloc_form`, which runs the eager
path. `kpp_surface_buoyancy` makes the same choice for the surface-relative
buoyancy behind Ritop.

## Sampled-input clocks

Time-updated components, boundary conditions and their nested providers declare
one frozen `SamplingRule` from `locomgcm.sampling`. Its `inputs` names the
sampled tensor attributes, `clock` is `SampleClock.START` (t), `END`
(t + dt_clock), or `INDEPENDENT`, and `providers` names nested participants.
Boundary applications declare `entry_clock=START` and `clock=END`; sponges use
END throughout, while wind and surface forcing use START. Initialization uses
the actual initial time. The common preparer resolves these declarations;
the stepper must not choose clocks by component name. Assembly rejects missing
declarations, including nested providers. Preparation rejects declared scalar
inputs or an END update with only undeclared host attributes, naming the
component and attributes; analytic producers must materialize tensor samples.
Timestamps and cache keys remain host metadata, not scientific clocks.

MITgcm anchors: `model/src/do_oceanic_phys.F:322-323` samples OBCS at
`myTime+deltaTClock`; `model/src/load_fields_driver.F:214,234` loads surface
forcing at the step start; `pkg/obcs/obcs_init_variables.F:386-401` applies
initial boundaries. `pkg/obcs/obcs_apply_uv.F:205-209` gates both west normal
faces with `maskW(Iobc+1)` and scales the outer copy by `uvApplyFac`. Later
dry-value cleanup must preserve that stored outer boundary value.

## Nonlinear free-surface time levels

With `nonlin_free_surf >= 1` and `select_r_star = 0` the step holds two surface
thicknesses, the grid hFac and `hFac_surf`. Entering step n the grid holds
h(eta^(n-1)) and `hFac_surf` holds h(eta^n). The grid takes h(eta^n) before the
free-surface solve (MITgcm `UPDATE_SURF_DR`, `forward_step.F:852`), and
`hFac_surf` takes h(eta^(n+1)) after the continuity (`CALC_SURF_DR`,
`forward_step.F:959`). Synchronous tracers step before the solve, staggered
tracers after the continuity, so each reads a different pair:

| Quantity | Synchronous tracers | Staggered tracers |
|---|---|---|
| 1/hFacC of the surface forcing and the tendency divergence | 1/h(eta^(n-1)) | 1/h(eta^n) |
| hFacC the SST restoring flux is generated with (`forcing_surf_relax.F:95-104,113-151`), as read by KPP and by the tracers | h(eta^(n-1)) | h(eta^n) |
| Rescale hFacC/hFac_surf of the tendency and its AB history | h(eta^(n-1))/h(eta^n) | h(eta^n)/h(eta^(n+1)) |
| recip_hFacNew of the implicit vertical solve | 1/h(eta^n) | 1/h(eta^(n+1)) |
| Advecting velocity | u^n, v^n and the previous continuity's w | corrected u, v and this step's continuity w |
| PmEpR of the real-freshwater salt forcing and of the KPP surface forcing | previous continuity's -EmPmR*maskInC, zero on a cold start | this step's -EmPmR*maskInC |

PmEpR follows `external_forcing_surf.F:158-166`: synchronous stepping lags it
one step behind EmPmR to match the volume change its tracers see, staggered
stepping sets it from the current EmPmR before the surface forcing. Only
synchronous stepping reads the carried PmEpR (`_pmep_r`), also on the first
step after a restart, and only its restart requires the key.

MITgcm generates the SST restoring flux in DO_OCEANIC_PHYS with the
step-start hFacC and, under staggered stepping, rescales it by
hFac_surf/hFacC, so the flux carries the hFacC of the tracer stage that
applies it. KPP reads the flux at that point, before the geometry update.
The tracers divide it by drF times the hFacC current when they run, which is
the same layer, so their restoring tendency does not depend on the layer while
KPP's surface buoyancy forcing does. The stepper passes the generation hFacC
to `SurfaceHeatSaltFlux.surface_heat_forcing` for KPP
(`ImplicitFreeSurfaceStepper._relaxation_surface_hfac_c`); the tracer
evaluation passes the grid's hFacC, which holds that layer when the tracers
run.

Derivatives keep the dependence of both thicknesses on eta, as MITgcm's
adjoint does (`initialise_varia.F:314-318` adds the surface update to
`hfac*_ad`; `forward_step.F:954-959` stores etaH before `CALC_SURF_DR`).
Without autograd the surface level is written in place. With autograd
enabled, `UPDATE_SURF_DR` and the checkpoint codec install it by tensor
replacement (`MaskSet.replace_surface_hfac`), because an operation of the
step may already have saved the old tensor for backward. The surface level
of the new tensor comes from the new layer, the levels below from
`MaskSet.fixed_hfac`: the tensors held before the first replacement,
latched again when something else rebinds the attributes (the gradient
window binding hFac as a control at a segment head). The grid's
`hfac_*_for` entries, held apart from its other runtime memos, are then
rebuilt from the new tensors (`StructuredCGrid.refresh_hfac_runtime`),
converted copies of another dtype or device included. The in-place path
does not reach converted copies: a consumer reads the current mask tensor
only when it requests the masks' dtype and device. Grids built by the
runtime hold the masks at the run's dtype and device, which the operators
request. The gradient window carries both levels in the segment heads, and
the segment-head install also replaces the pressure coefficients and the
hFac entries built at preparation. A linear free surface has no such
install. There an hFac binding is accepted, but the backward of the second
segment fails with "Trying to backward through the graph a second time":
the binding (`checkpoint/recompute.py` `_bind`) leaves the prepare-time
hFac entries and first-assembly pressure coefficients in place, the
linear free surface saves no nonlinear payload whose install would replace
them, and the window does not reject the binding beforehand. hFac-derived
quantities (recip_hFac, corner hFacZ) are
recomputed from the current tensor on every call while the geometry is
time-varying or carries a graph, whatever `LOCOMGCM_CACHE_CONSTANTS` is.

The thickness takes MIN over the cells of a face or corner, so a central
difference along an initial-eta direction is smooth only while no pair of
those cells swaps order within the step size. The swap distance computed
from the initial eta and the direction bounds the step at the first step
only: along the trajectory the pairs move, and the hFacInf floor, the
periodic seam and the open-boundary faces add branches, so a step below
that distance is not shown to stay on one branch for the whole window.
Derivative checks pick the step below that distance, or a direction that
moves both cells equally, and check that two steps agree.

MITgcm's CG2D adjoint passes the coefficient cotangents only with
`cg2dFullAdjoint=.TRUE.` (`cg2d_mad.F`, under `NONLIN_FRSURF` or
`ALLOW_DEPTH_CONTROL`); the default `.FALSE.` (`autodiff_readparms.F`)
zeroes them. With nlfs=3 the matrix follows hFac every step
(`forward_step.F`, `UPDATE_CG2D`), so the default adjoint omits the
eta-to-matrix term that LOCOMgcm's reverse mode keeps; nlfs=1 and 2 build the
matrix once from the resting geometry, which does not depend on eta but
does depend on a topography control.
