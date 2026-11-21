# Implicit convective KPP execution

The optional stepper mapping `kpp_convective_implicit` selects
`convective_be_column_v1`. Setting the mapping to `null` retains the lagged
KPP scheme. The implicit scheme supports one regional latitude-longitude tile,
batch size one, float64 tracers, JMD95 KPP buoyancy, and implicit vertical
advection of both temperature and salinity. Nonlocal and continuity terms
remain outside Adams-Bashforth.

## Equation and derivative

The nonlinear station solves the existing coupled temperature/salinity
residual at each horizontal column. New tracer values affect the raw buoyancy
jump and its convective diffusivity through `RI_IWMIX`, `BLMIX`, `enhance`, and
`combine`. Diagnosis-time shear, smoothed Richardson number, boundary-layer
depth/index, pressure, vertical velocity, and surface forcing are external
inputs. Their complete dependence on the outer model remains in the automatic
external-input pullback. The station evaluates only STATEKPP's adjacent-layer
buoyancy jump, using the same JMD95 density and divided differences; unused
surface-density and expansion-coefficient outputs are not constructed.

The physical state is represented by scaled increments
`Y = Y_old + scale * z`. `root_tolerance` bounds the infinity norm of the
original scaled residual, evaluated again on the rounded physical state that
the station actually returns. The implicit backward recomputes this residual,
solves the transposed scaled Jacobian, and applies the complete residual VJP.
The transpose equation has a relative residual bound of `1e-11`. Root and
adjoint checks are independent of the execution backend.

Newton uses the existing lagged solution as its initial guess, Armijo
backtracking, and the configured number of polishing passes. The existing
rank-one, factor-box, and factor-homotopy policies remain available. Each
column retains its first accepted line-search state together with that
state's residual for the next Newton decision. Already converged columns
retain their previous pair. The final rounded physical state is still checked
with a fresh evaluation of the original equation. Polishing accepts only an
improvement that also meets the root tolerance. Its five independent trial
states share one column-batched residual evaluation in each pass, with the
original strict power order retained for selection and equal-norm ties.
Repeated geometry and auxiliary inputs belong only to that polishing call
and are not retained with the accepted root. A retained rank-deficient
recovered root has no implicit-function pullback and is rejected in backward.

When a column exhausts its line search, Newton retains the accepted updates
of the other columns and continues them within the same iteration budget.
Stalled columns remain frozen for recovery. After the first stall, linear
solves select only active columns; residual and acceptance programs retain
the fixed station shape. A station without a stalled line search follows
the original computation and record path.

Recovery groups failed columns by their number of effective convective
interfaces. Scalar Brent and factor-box Newton run on the current device,
with independent convergence and line-search masks. The factor-box start
order, duplicate-start suppression, finite differences, and complete-residual
acceptance remain the same. Candidate batches bound temporary dense storage.
The compiled path fuses the fixed-factor coefficient construction, conservative
matrix checks, and physical linear solve in one device program. Factor
residual reductions and per-column decisions remain outside that program.
Only columns still unresolved by both stages move to CPU for homotopy.
Polishing, the rounded-state residual check, and the recovered Jacobian's
singular-value diagnostics use the original device path. Recovery records
retain their per-column fields; a concurrent group's elapsed time is reported
in each member's `wall_s`, while `recovery_wall_s` measures the entire station
once. A solve that converges before recovery uses the original Newton path.

## Compact exact Jacobian

A column's Jacobian is not merely block tridiagonal. In scaled coordinates it
has the exact form

```
J = B + U V^T
```

`B` is a 2 by 2 block tridiagonal matrix, with temperature and salinity paired
at each vertical level. It contains linear transport and all local convective
cross-tracer derivatives outside the boundary layer. Boundary-layer matching
reads the three convective interfaces `kn-1`, `kn`, and `kn+1`; the interface
read by `enhance` belongs to this same set for both matching cases. `U V^T`
propagates those dependencies through the complete boundary layer. Its rank is
at most three.

Assembly preserves the original analytic derivative's masks, inclusive clamp
convention, `abs'(0)=0`, and half-weight derivative at maximum ties. It never
forms a full field of dense `2*nz` by `2*nz` matrices. Coefficients and work
arrays have storage proportional to vertical levels times physical columns.
The reusable block solver lives in `locomgcm.solvers._block_tridiag`; its
CUDA implementation keeps independent columns contiguous and performs each
vertical recurrence in one Triton kernel with float64 arithmetic and separate
multiply/add rounding.

For `W = B^-1 U` and `M = I + V^T W`, the two orientations use

```
x      = B^-1 r - W M^-1 V^T B^-1 r
lambda = B^-T (r - V M^-T W^T r)
```

Block factors serve every right-hand side in that linear solve. The transpose
path uses the same algebra and supports multiple right-hand sides. Every
column's linear equation is checked after solving. A failed block pivot does
not imply that the full Jacobian is singular: only failed columns fall back
to the original analytic dense matrix, in chunks of at most 64 columns. Dense
factors and their operators are retained only for the lifetime of that solve.
Failure of the original dense equation raises the station's diagnostic
`RootFailure` rather than accepting an unchecked correction.

The Newton matrix generally belongs to the iterate before an update. Its
factorization therefore cannot be reused blindly for the final rounded
physical state. Backward assembles the compact exact Jacobian at the returned
state and factors it there.

## Column and checkpoint ownership

`compact_columns` restricts the nonlinear station to columns containing at
least one wet cell. The vertical extent remains fixed. Fully dry columns have
the exact equation `Y=rhs`; differentiable gather/scatter outside the implicit
boundary preserves this identity and its RHS cotangent. Their RHS must still
be finite, including for a completely dry domain. Spatial KPP operands
follow the same selection, while vertical coordinate and lookup tables retain
their original axes. Newton residuals keep a fixed wet-column shape across
iterations, including after a stalled line search.

With `checkpoint_reuse`, a checkpoint segment retains an owned, detached copy
of each accepted physical root and its Python diagnostic metadata. It stores
no problem object, factorization, or autograd graph. Root tensors are attached
to the existing checkpoint slot, interleaved with pressure roots in their
original per-step order. The existing scheduler budgets, places, transfers,
and releases all these bytes together with the segment head.

Recompute consumes the retained roots in order and skips the lagged starting
solve, Newton, line search, and polishing. It still evaluates the original
residual and enters the ordinary implicit autograd boundary. Shape, dtype,
device, equation options, root count, and the persistent grid-mask identity
and version must agree. The version check requires an ordinary tensor when
root reuse is active. Normal grid preparation creates ordinary persistent
constants even when the forward kernel runs in inference mode. A caller that
constructs its own inference grid must prepare ordinary constants or disable
root reuse.

Outside a checkpoint segment, root retention is inactive and creates no
snapshot. Recompute and failure release consumed roots and metadata; window
termination releases their store slots. The window retains its existing
first-order backward contract. The scheduler and the store are
`locomgcm/checkpoint/window_schedule.py` and `locomgcm/checkpoint/store.py`.

## Execution options

The three infrastructure options are independent of the physical equation:

| Option | Default | Meaning |
|---|---|---|
| `linear_solver` | `structured` | Use the compact exact Jacobian; `dense` selects the original full matrix. |
| `compact_columns` | `true` | Exclude fully dry columns from the nonlinear station. |
| `checkpoint_reuse` | `true` | Retain accepted physical roots in checkpoint segments. |

For example, within the stepper configuration:

```yaml
kpp_convective_implicit:
  execution: compiled
  linear_solver: structured
  compact_columns: true
  checkpoint_reuse: true
  root_policy: newton_factor_homotopy_v1
  root_tolerance: 1.0e-11
  polish_steps: 2
```

The compiled residual, compact assembly, device solve/check, and tensor
control programs require `TRITON_DEFAULT_FP_FUSION=0`. The solve/check program combines factorization,
Woodbury application, and the full relative-residual check before returning
one two-number host summary. It preserves factor ownership and keeps selected
dense fallback outside the compiled program. CUDA block factors use the native
backend; CPU factors use the torch implementation. The host root controller retains
its explicit compile boundary because iteration acceptance, numerical
failures, and fallback routing depend on evaluated residuals. Newton summaries,
line-search acceptance, and strict polishing selection use fixed compiled
tensor programs within that boundary. The vector norms used by Armijo and
strict polishing comparisons retain their original ATen reductions outside
these programs: compiling those reductions can change equal-norm selection
near the rounding limit. The host receives the same decisions and preserves
the original first-accepted state, power order, failure records, and checks.
Each line-search read also carries the next Newton summary, so the controller
does not wait again before the next iteration. Provisional summaries exclude
unaccepted columns; they are used only when every active column has accepted
or when the final trial has established the original stalled set. This retains
the original residual evaluations, iteration history, and checked linear solves.

Compiled scalar recovery groups retain the original Brent endpoint and pending
column selection. Three fixed device programs combine the original bracket
update, interpolation proposal, and evaluated-state commit expressions.
Compiled execution reads the device activity mask once after the bracket
update. An inactive bracket update preserves the selected factor, evaluated
state, failure code, and diagnostic iteration count, so it also handles
inactivity from the preceding commit without a separate entry read. The
original fixed-factor evaluations, dense solve, and all checks are unchanged.
Eager execution retains its original two reads per iteration. Diagnostic iteration
numbers use a once-prepared device integer table and zero-dimensional views;
the table is prepared once on entry to the compiled loop. Each
compiled program receives the current integer value without specializing on
a Python iteration number. Eager recovery uses the same original expressions.
The programs compile with dynamic shapes so a different scalar-column count
does not require one specialized program per station.
The compiled proposal receives the original `-fcur` from the host controller
as a Tensor. This unary operation stays outside the fused program because
the current Triton lowering uses `+0 - fcur`, which changes the sign of a
zero result. Both trial expressions reuse the supplied negation; their
multiplication, division, and selection order stays unchanged. Eager
recovery retains its original unary expressions. This boundary can be
removed when the compiler preserves unary-negation signed-zero semantics.

The reference configuration is `linear_solver: dense`,
`compact_columns: false`, and `checkpoint_reuse: false`. It is useful for
paired comparisons using identical inputs, acceleration settings, dtype,
root policy, scaling, and tolerance.

## Measurement contract

`statistics()` distinguishes actual Newton stations, replayed stations, and
backward stations. It reports Newton and line-search work, polishing work,
full and solved column counts, compact and dense linear solves, exceptional
dense columns, and maximum primal and adjoint residuals. Replayed stations do
not count the original Newton history or recovery work a second time.

Assess a change using full forward steps and complete checkpointed
value-and-gradient windows. Include checkpoint transfer, recompute, residual
VJPs, host synchronization, compilation, and peak memory in the relevant
measurements. Report compilation and diagnostic instrumentation separately
from warmed steady-state timing. A local linear-solve improvement alone does
not establish an end-to-end speedup.
