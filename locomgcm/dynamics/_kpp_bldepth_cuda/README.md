# Native boundary-layer depth reverse

`LOCOMGCM_KPP_BLDEPTH_CUDA=1` selects the existing native forward and its first-order
reverse. The forward CUDA arithmetic is unchanged. The reverse launches one
thread per column and scans the levels once; it does not build a full-volume
select-scatter for each level.

The admitted differentiable fields are `dvsq`, `dbloc`, `Ritop`, `ustar`, `bo`,
`bosol`, and `coriol`. The continuous outputs `hbl`, `bfsfc`, and `Rib` accept
independent cotangents, including absent cotangents. `kbl`, `stable`, and `casea`
are discrete outputs. Geometry, lookup tables and Python scalar parameters are
fixed in this implementation. Active geometry or tables decline with
`geometry_grad`; tensor parameters and velocity-scale overrides
retain their existing decline reasons. The public dispatcher records a decline
and uses the reference. The strict direct entry raises. Forward-mode transforms
decline with `torch_transform`, and higher-order reverse requests raise.

Clamp passes its input derivative at equality. Minimum splits equal-operand
cotangents in half. Abs has zero derivative at zero; the guarded square root
has zero contribution when buoyancy frequency is zero. Branch conditions and
the lookup-table integer indices have zero derivative. Both signed-zero modes
and the penetrating-shortwave switch follow the reference's branches.

The ignition ledger records `variant=forward` and `variant=backward` at the
custom-op execution boundary. Loading or tracing the fake op records no hit.
Selected execution errors propagate; reverse errors also record their exception
class. Direct calls while the registry switch is disabled do not populate its
ignition ledger.

The implementation table (`locomgcm/accel/table.py`) retains `NOT_RUN` for
unqualified execution combinations.
