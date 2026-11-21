# Numerical validation

Compare the same discrete mapping and complete declared derivative dependencies.
The shape and cause of a difference matter, not its magnitude alone. A sign,
branch, mask, variable, time-level or dependency mismatch is a defect even when
its absolute size is near the floating-point floor. Unstructured differences
caused by summation order, fused multiply-add or reciprocal multiplication are
assessed against the operation's conditioning and the actual comparison scope.

## Establish the reference

Before comparing with MITgcm, record:

1. The build identity, including CPP options, packages, compiler and flags.
2. Actual namelist values, including package and boundary settings.
3. Enabled algorithms and switches, including staggering, forcing placement,
   free-surface pressure, nonlinear free surface and vorticity scheme.
4. First-step checkpoints with variable names, shapes, masks, units and conventions.

Use the reference model's actual runtime values, not assumed defaults. Capture
reference states at meaningful time-integration boundaries and follow the first
unequal checkpoint. Do not infer individual terms from a mixed total or cycle
through speculative operators without reference observations.

Fortran stream dumps require an explicit byte order and shape. A reference built
with big-endian conversion needs corresponding `>f8` reads for double precision.
Record the input and output paths.

An investigation states its claim, required source and runtime evidence, pass/fail
condition, and the condition that closes that claim. Check comparison scripts for
step pairing, masks, precision and interpolation before trusting their summaries.
Important conclusions need independent evidence paths; two reports using the same
suspect parser are not independent verification.

## Interpret comparisons

Retain absolute and relative differences, locations, dtype, branch choices,
active inputs and output cotangents. Near zero, preserve absolute differences
rather than choosing an arbitrary denominator. Unexplained differences remain
unverified; do not relax thresholds to make a candidate pass.

Bitwise agreement is required for an externally reported LOCOMgcm–MITgcm
bitwise alignment conclusion and for a claim that a change is numerically neutral.
Accelerated and derivative implementations otherwise use algorithmic-equivalence
acceptance. Strict execution requires bitwise reductions only when their order is
controlled. FMA switches can diagnose contraction effects; diagnosed bounded
floating-point differences are not separate implementation defects.

Established case contracts retain their defined scope. Diagnostic scales from an
earlier kernel or dtype are not universal acceptance thresholds. Define any new
bound from theory, reference noise and error-detection ability before evaluating
the candidate under it.

A magnitude gate alone cannot reject a structured difference below it, so a
comparison against an independent reference also judges structure. In a
gradient comparison with the MITgcm adjoint, a constant factor and a constant
offset are fitted to LOCOMgcm - MITgcm over each group of wet cells (the connected
cells, the open-boundary ring, the boundary band and the mask edge). A scale
error or a uniform shift that dominates the group's residual makes the fit
explain nearly all of its energy, however far it lies below the 1e-10 gate, so
a group is flagged when the fit explains more than half of that energy or a
cell where both are non-zero changes sign, at any magnitude. A confirmed
algorithmic attribution fails a group whether or not it is flagged. A flag is
not a conclusion. The same share, the same size of the fitted part and the
same number of cells carrying the residual
arise from a legitimate reordering of one sum in every cell and from an
injected offset; grid cells are not independent samples, and no fixed number
of units in the last place or of cells separates the two. A flagged group is
therefore decided only by an attribution of the protocol above: a named
operation whose rounding bound is established from its accumulation length,
cancellation and propagation passes the group when its largest difference is
within that bound, and a confirmed algorithmic difference fails it. Without an
attribution the group waits. A comparison report can also state for every
compared station whether its largest differences cluster within two cells of a
dry cell or the array edge; such a report accompanies the verdict and does not
decide it.

A bitwise comparison compares stored bits: the same files, variables,
coordinates and dtypes, and zero elements whose bits differ, so -0 differs from
+0 and NaNs differ by sign and payload. Equal numerical values are not a bitwise
result. Variable attributes are compared as strictly as values (an equal value
of another type differs), since the same writer produced the reference; a
writer change that alters them is reviewed and accepted as a new reference.

A timing CV limit states whether the samples can separate a regression from
normal variation. Above it the measurement is inconclusive: it neither passes
nor fails, blocks a release and is repeated, not accepted.

## Derivatives and execution

Identify whether differentiation applies to an equation's solution or to a finite
iteration program. Check smooth-region directional differences, Taylor behavior,
equation residuals or independent derivatives as appropriate. Adjoint dot-product
self-consistency alone cannot exclude a shared error.

For a complete integration window, include initial state, all carried history and
all controls reaching the objective. Validate requested combinations such as JVP,
VJP, compilation, recomputation and distributed execution in their actual scope.
Verify that requested native kernels executed; a correct fallback is not evidence
that the accelerated implementation ran.

Performance comparisons use a valid complete gradient or forward path under the
same configuration. Include conversion, copies, memory traffic, allocation,
saved state, recomputation, launches, synchronization and communication. Record
repeated warm costs and peak memory; distinguish measured improvement from normal
variation. Hardware-specific and local checks support only their measured scope.

## Release verification

A release names the checks it requires. Each required check must execute on
the host of the run and give PASS against a reviewed reference. A planned,
skipped, unreviewed or diagnostic check, a command failure, a failed or invalid
criterion, an inconclusive timing and a missing metric or file each block the
release; the success of a diagnostic run does not count as a release check.

A check of the gradient path is accepted only together with its negative
control: a deliberately broken variant of the same measurement that the check
must reject, run in the same check. A check that accepts its negative control
has not shown that it can detect the defect it targets. GPU checks record where
segment heads were kept and the device memory in use inside and outside the
caching allocator, so a resource claim states its placement and its
measurement point. On a shared GPU a claim about one process's memory uses that
process's own figure; a check that cannot attribute the memory is incomplete,
and an incomplete check fails.

A check of numerical-failure handling injects a real fault into the solver and
observes every participating process on its own: its exit status, its record
and whether parameters changed. A process ended by a launcher, a timeout or a
signal fails the check. The contract checked is the numerical-failure channel
of `locomgcm/deferred_checks.py`: without a communication session the failure
raises `DeferredCheckError` where it is observed, and inside a session
(`locomgcm/distributed/comm_adjoint.py`) every rank raises `CommSessionError`
carrying the diagnostics of every failing rank.
