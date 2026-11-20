# Using LOCOMgcm

This guide covers running the built-in `pureocean` case on a CPU, driving the
model from Python, preparing your own case, and enabling GPU kernels. Commands
run from the repository root unless stated otherwise. Everything below was run
with Python 3.11, torch 2.9.0 and the versions in `requirements-runtime.txt`.
Statements marked *unverified* were read from the code but not run.

## Quick start on CPU

### Install

```sh
git clone <repository-url> LOCOMgcm
cd LOCOMgcm
python -m pip install torch==2.9.0          # or the CUDA build for your machine
python -m pip install -r requirements-runtime.txt
python -m pip install --no-deps -e .        # or: python -m pip install --no-deps .
python -c 'import defgcm; print(defgcm.__file__)'
```

Use torch 2.9.0. torch 2.5.1 fails at import with
`AttributeError: module 'torch.compiler' has no attribute 'nested_compile_region'`.
Versions between 2.5.1 and 2.9.0 were not tried.

If torch and the packages in `requirements-runtime.txt` are already installed,
the first two `pip` lines report `Requirement already satisfied` and change
nothing. `python -m pip install .` without `--no-deps` also works but may
replace an existing torch with the pinned 2.9.0.

Every process that imports `defgcm` prints 28 `DEFGCM_ACCEL {...}`
lines to stderr at exit, including the import check above (`--help` of the
pipeline tools returns before the import). They report whether each
accelerator switch ran (see "Switches"). Set `DEFGCM_ACCEL_REPORT=0` to turn them
off.

The wheel contains `defgcm`, `experiments` (with all case files under
`experiments/configs/cases/`), `helper` and `contracts`. A non-editable install
leaves `build/` and `defgcm.egg-info/` in the checkout; delete them before
building again. If more than one copy of LOCOMgcm is installed, check which
one Python imports with the command above.

Packages imported by optional parts of the code but missing from
`requirements-runtime.txt` are `xskillscore` (stage 4 comparison metrics; the
LOCOMgcm-only stage 4 does not import it), `pyTMD` (tidal boundaries),
`scores`, `psutil`, `threadpoolctl` and `omegaconf`. Stages 1a, 1b, 2 and 4
of `pureocean` ran with all of these absent.

### Create inputs

The `pureocean` case (36 x 36 x 10 cells, 1/12 degree, dt = 120 s, 7-day
horizon) reads HYCOM ocean state, ERA5 10 m winds and GEBCO bathymetry in
stage 1a. None of these ship with the repository. An unconfigured data key
raises `ExternalPathError` while loading the case. Write synthetic
files in the same formats:

```sh
export DEFGCM_SYNTHETIC_PUREOCEAN_DIR="$PWD/data/synthetic_pureocean"
python examples/make_synthetic_inputs.py --out "$DEFGCM_SYNTHETIC_PUREOCEAN_DIR"  # 28 MB
```

The synthetic fields have a flat 1000 m bottom with no land, depth-dependent
temperature and salinity with a warm anomaly in the middle of the domain, zero
initial velocity and surface height, and a uniform easterly wind with a daily
modulation. They exist only to exercise the code. `data/` is git-ignored.

### Run the pipeline

```sh
python -m experiments.pipeline_v2.run_pipeline \
  --config experiments/configs/cases/pureocean.yaml \
  --overlay examples/pureocean_cpu.yaml \
  --run-id pureocean_cpu
```

`examples/pureocean_cpu.yaml` sets `run.device: cpu`, points the three
`sources` at the synthetic files, runs 60 steps and writes a NetCDF frame every
20 steps. pureocean has `mit_comparison.enabled: false`, so without `--stages`
the run is stages 1a (prepare inputs), 1b (LOCOMgcm inputs), 2 (LOCOMgcm)
and 4 (postprocessing); the log says so in its first lines. From the first
log line to the last took 10 s: stage 1a 6.0 s, stage 1b 0.8 s, stage 2
2.5 s, stage 4 0.3 s. The orchestrator writes nothing to stdout; its log lines
and the `DEFGCM_ACCEL` report go to stderr. Stage 2's own output goes to
`_pipeline_meta/stage2_torch_run/stage2_stdout.log`:

```
[3/4] Running LOCOMgcm (60 steps from global step 0, dt=120.0s) ...
  step     0/60  t=    0.0h  η_rms=0.0000e+00  KE=0.0000e+00
  step    50/60  t=    1.7h  η_rms=6.3738e-03  KE=5.9347e+08
  step    60/60  t=    2.0h  η_rms=6.7091e-03  KE=8.3175e+08
  LOCOMgcm runtime: 2.8s
  Captured 4 LOCOMgcm snapshots
```

Required details:

- Without `--stages`, the stage set follows `mit_comparison.enabled`: false
  gives 1a, 1b, 2 and 4; true gives every stage. `--stages all` means every
  stage, including 1c (MITgcm input export) and 3 (MITgcm build and run),
  which need `mit_comparison.enabled: true` and a configured MITgcm tree
  (`docs/INPUTS.md`); with the comparison off, a stage list that names 1c, 3
  or 5 is refused before any stage starts (`the MITgcm stages ... need
  mit_comparison.enabled=true`, exit code 5).
- The case file sets `run.device: cuda`. On a machine without a GPU, stage 2
  fails with `No CUDA GPUs are available` unless the device is `cpu` or `auto`.
- `--output DIR` moves the output root (default `output.root` in the yaml,
  `outputs/data`). Relative and absolute paths both work.
- `DEFGCM_PATHS` is not needed for stages 1a, 1b, 2 and 4 of this case. It
  configures MITgcm and CS32 locations only.

`python -m experiments.pipeline_v2.show_run_status pureocean_cpu --output outputs/data`
prints the stage table. Its positional argument is the run directory name,
which is `run.experiment_name`, not the `--run-id` value. Without `--output`,
it looks next to the installed package.

### Where the output is

With `output.legacy_layout: true` (the default), the run directory is
`<output>/<run.experiment_name>`. The `--run-id` value is recorded in
`manifest.yaml` but does not name the directory. Set `run.experiment_name`, as
the overlay does, to choose the directory. *Unverified:* with
`legacy_layout: false`, the run id names the directory.

```
outputs/data/pureocean_cpu/
  manifest.yaml                stage status and artifact list
  metadata.yaml, provenance.yaml
  config_snapshot/             written by stage 1a only: source_config.yaml (the case
                               file as given, overlays not merged), resolved_args.yaml
                               (the merged configuration stage 1a used), git_state.txt
  nc_snapshots/day001.nc       LOCOMgcm state frames (NetCDF), one file per model day
                               with the default intervals (see below)
  preprocessed.zarr/           stage 4 output (see "Postprocessed output")
  _pipeline_meta/
    preprocessed_cf/           prepared_{state,forcing,boundary,mesh}.zarr (stage 1a)
    model_input/               initial_state.zarr, grid, run_params.yaml (stage 1a)
    torch_inputs/              torchocean_obc.zarr (stage 1b)
    stage2_torch_run/          stage2_stdout.log and stage2_config.yaml (the case file,
                               overlays, --set items and merged configuration of the
                               latest stage 2 run)
outputs/data/_cache/           cache.root, relative to the working directory
```

The stage 1b consumers and their files are named `torchocean_ic_zarr`,
`torchocean_obc_zarr`, `torchocean_forcing_zarr` (`torchocean_obc.zarr` and so
on), and their bundle metadata store the sign-convention label `torchocean`,
written in case files as `to: torchocean`.

`nc_snapshots/dayNNN.nc` holds `eta(time, lat, lon)` and
`temp, salt, u, v, w(time, depth, lat, lon)` in float64, all on the `nx` by
`ny` cell grid. `u` is the west-face value of each cell; the model's extra
east boundary face (`nx+1` faces) is not written. `v` is the south-face
value. `w` is upward-positive at the top face of each cell. The coordinates
need care:

| Coordinate | Content |
|---|---|
| `time` | hours since the start; the attribute is `time_unit: hours`, not CF `units` |
| `depth` | cell-centre depth in m (50, 150, ..., 950) |
| `lat` | cell-centre latitude in degrees, `center_lat + (j - (ny - 1) / 2) * resolution_deg` (10.2917 to 13.2083 here) |
| `lon` | cell-centre longitude in degrees, same formula with `center_lon` (110.2917 to 113.2083 here) |

`lat` and `lon` are degrees when `case.extras.grid_mode` is `spherical_polar`.
On a Cartesian grid (`grid_mode` absent or `cartesian`) `lat` is the row index
0 ... ny-1 and `lon` the cumulative sum of the cell widths along the first row
in m. Variables have no `units` attributes.

Two fields look like they set the output cadence.
`output.snapshot_interval_seconds` is the interval between LOCOMgcm frames:
600 s gave 13 frames over the 2-hour run, 2400 s gave 4.
`output.nc_output_interval_seconds` is the MITgcm dump interval. On the
LOCOMgcm side it only sets how the frames are split into files: a new file
starts every `int(86400 / nc_output_interval_seconds) * nc_output_interval_seconds`
seconds, which is one model day when the interval divides 86400 (2400 s and
600 s do; 6000 s started `day002.nc` at 23.33 h). Stage 2 prints the effective cadence as
`[output cadence] TO writes a NetCDF frame every N steps` and warns when the
two differ, because the MITgcm comparison expects them equal.

```python
import xarray as xr
ds = xr.open_dataset("outputs/data/pureocean_cpu/nc_snapshots/day001.nc")
print(ds.time.values)                 # [0. 0.6667 1.3333 2.]
print(float(abs(ds.u).max()))         # 0.03055 m/s at the last frame
```

### Postprocessed output (stage 4)

With `mit_comparison.enabled: false`, stage 4 reads `nc_snapshots/*.nc` and
the stage 1a grid tensors and writes `preprocessed.zarr`. Group `model` holds
`temp_spatial`, `salt_spatial`, `u_spatial`, `v_spatial`, `w_spatial`
(time, depth, lat, lon) and `ssh_spatial` (time, lat, lon) in float32;
`timeseries/hfac.npy` is the wet-cell mask; `_SUCCESS` is written last, so a
store without it is incomplete. The store attributes carry the coordinates:
`lat` and `lon` are cell centres in degrees, `center + (i - (n - 1) / 2) *
resolution_deg` (here 10.2917 to 13.2083 and 110.2917 to 113.2083), `depth`
is cell-centre depth in m, `time` is seconds since the start (0, 2400, 4800,
7200). The arrays are the NetCDF fields rounded to float32: `temp_spatial`,
`salt_spatial`, `w_spatial` and `ssh_spatial` sit at the cell centres the
coordinates describe, while `u_spatial` and `v_spatial` are still the west-face
and south-face values, half a cell west and south of them. `manifest.yaml`
records the store with `extras: {mode: model_only}`. Listing the store's keys
makes zarr print `Object at _SUCCESS is not recognized` and the same for
`timeseries`; both are harmless.

```python
import numpy as np, zarr
z = zarr.open("outputs/data/pureocean_cpu/preprocessed.zarr", mode="r")
lat, lon = np.array(z.attrs["lat"]), np.array(z.attrs["lon"])
sst = z["model/temp_spatial"][-1, 0]   # last frame, top layer, (36, 36)
```

With `mit_comparison.enabled: true`, stage 4 needs the MITgcm output of stage
3, adds `mitgcm` and `diff` groups, and its metrics import `xskillscore`.
Stage 5 compares the two models.

## Using the model from Python

Both examples call `examples/_pureocean.py`. On first use it writes the
synthetic inputs and runs stages 1a and 1b under `outputs/examples/`. Later
runs reuse them. Use `--work DIR` to choose another location.

### Forward run: `examples/minimal_run.py`

```sh
python examples/minimal_run.py            # --steps 20 by default
```

Observed output (inputs already prepared, total 3.6 s). The run starts with
eight `DEFGCM_ACCEL_PRELOAD ... status=skipped reason=not_requested`
lines, one per native kernel that was not requested, and the monitor prints
every 5 steps:

```
  step     0/20  t=    0.0h  η_rms=0.0000e+00  KE=0.0000e+00
  step     5/20  t=    0.2h  η_rms=6.7261e-03  KE=2.1501e+07
  step    10/20  t=    0.3h  η_rms=1.0623e-02  KE=3.0346e+07
  step    15/20  t=    0.5h  η_rms=9.6638e-03  KE=5.9428e+07
  step    20/20  t=    0.7h  η_rms=6.7602e-03  KE=1.0557e+08
grid 36 x 36 x 10, dtype torch.float64, device cpu, dt 120 s
after 20 steps (0.67 h):
  max |u| = 1.1309e-02 m/s, max |v| = 1.0398e-02 m/s
  area-mean SST = 24.536589 degC
  max |eta| = 3.1498e-02 m
wall time: build 0.87 s, 20 steps 0.54 s
```

The core calls are below. The snippet was run as written after the quick
start had produced `outputs/data/pureocean_cpu`:

```python
from pathlib import Path

from defgcm import ModelRunner
from defgcm.io.runner.config import load_run_config
from experiments.cases import build_model

steps = 20
run_cfg = load_run_config(
    "experiments/configs/cases/pureocean.yaml",
    ["examples/pureocean_cpu.yaml"],          # overlays
    [f"time.steps={steps}"],                  # --set items
)
run_dir = Path("outputs/data/pureocean_cpu")  # written by stages 1a and 1b
model, state, dt = build_model(run_cfg, run_dir, steps=steps, device="cpu")
result = ModelRunner().run(
    model, state, steps=steps, dt=dt, snapshot_steps=[steps], collect_snapshots=False,
)
final = result.final_state
print(f"max |u| = {float(abs(final.u.data).max()):.4e} m/s")   # 1.1309e-02
```

The run prints eight `DEFGCM_ACCEL_PRELOAD ... skipped` lines, the monitor
every 2 steps, then `max |u| = 1.1309e-02 m/s`. `build_model` performs the
assembly stage 2 does before stepping and returns the model, its initial state
and the per-equation time-step bundle. State tensors carry a leading axis of
length 1:

| Field | Shape | Location |
|---|---|---|
| `final.eta.data` | `[1, ny, nx]` | cell centre |
| `final.temp.data`, `final.salt.data` | `[1, nz, ny, nx]` | cell centre |
| `final.u.data` | `[1, nz, ny, nx+1]` | west faces |
| `final.v.data` | `[1, nz, ny+1, nx]` | south faces |

Masks and metrics are on `model.grid`, for example
`model.grid.hfac_c_for(dtype, device)` and `model.grid.metrics.cell_area`.

Build a new model for every run. A model keeps its time-stepping history
after `ModelRunner.run`. A second run of the same model object from the same
initial state differed from a fresh model by up to 3.1e-3 K in temperature and
5.5e-4 m/s in velocity after 10 steps. A freshly built model reproduces the
first run exactly. Building pureocean takes under 1 s.

### Gradient: `examples/gradient_example.py`

The objective is the mean top-layer temperature in the central 8 x 8 cells
after N steps. The script replaces the initial temperature with a leaf that
requires gradients, runs `ModelRunner.run(..., track_gradients=True)` and calls
`torch.autograd.grad`. It then compares one directional derivative with a
central finite difference along a random non-negative direction over wet
cells.

```sh
python examples/gradient_example.py       # --steps 10 --h 1e-6 by default
```

`--save DIR` also writes the gradient to `DIR/gradient.npy` and J, the two
directional derivatives and their relative difference to `DIR/metrics.json`.

Observed output:

```
J = 25.021946966682 degC (top-layer box mean after 10 steps)
|dJ/dT0|_2 = 1.242474e-01, sum = 9.999105e-01, top-layer box sum = 9.937762e-01
nonzero gradient entries: 11560 of 12960
directional derivative: AD = 5.101441085974e-01, FD = 5.101441100663e-01, relative difference = 2.880e-09 (h = 1e-06)
wall time: forward+backward 1.50 s, two FD forwards 1.48 s
```

The full step differentiates on CPU in float64. The 1400 zero entries are
exactly the outermost ring of cells in every layer, where the open boundary
prescribes temperature. The derivative holds the model's reference
temperature and salinity profiles fixed, because they are computed from the
prepared initial state when the model is built.

Upwind advection makes the objective piecewise smooth. A finite-difference
step that crosses a branch change agrees only to about 1e-5. Observed
relative differences between the two derivatives:

| steps | h = 1e-3 | h = 1e-4 | h = 1e-5 | h = 1e-6 |
|---|---|---|---|---|
| 10 | 4.5e-5 | 9.5e-10 | 2.0e-9 | 2.9e-9 |
| 40 | not run | 3.6e-5 | 6.0e-5 | 1.3e-8 |

### Forward-mode derivatives

A directional derivative (JVP) is taken by making a control a dual tensor
inside `torch.autograd.forward_ad.dual_level()` and running
`ModelRunner.run(..., track_gradients=True)` uncompiled, on one rank, without
a checkpoint window. That is the verified scope: CPU, float64, eager, against
central differences and the reverse mode of the same control, for the
free-surface (CG2D) and nonhydrostatic (CG3D) solvers alone and for whole
hydrostatic runs with a linear free surface (the nonlinear one: see the last
paragraph of this section). GPU execution has not been verified. Neither have the `torch.func`
transforms (`jvp`, `jacfwd`, `vmap`) nor nesting forward over reverse: none of
them is part of what the model provides, although the checks below do read an
open `torch.func.jvp` level as live forward mode.

The pressure solves differentiate the solution of their equation (ROOT), as
their reverse mode does: the tangent is `x_dot = A^-1 (b_dot - A_dot x)`, one
more solve of the same system with the reverse re-solve's relative tolerance,
and the first guess and the preconditioner carry no derivative (MITgcm
`CG2D_B0`). This is what the model's solves take: CG2D through `cg2d_solve`
(which passes the coefficient tensors as `root_operands`), CG3D with
`root_adjoint=True`, or in its automatic mode with a converging iteration
budget and no derivative through the first guess or the preconditioner. The
other choices differentiate the finite program, the iterations actually
executed, first guess and preconditioner included: `root_adjoint=False` and,
in CG3D's automatic mode only, an iteration budget below the ROOT minimum or a
tangent or gradient on the first guess or the preconditioner. With
`root_adjoint=True` CG3D takes ROOT whenever a derivative is requested through
any input or forward mode is live, whatever the budget: a solve that does not
converge raises, and a tangent on the first guess alone gives zero.

The accelerator gates decline while forward mode is live (an open dual level,
except inside the forward of a custom `autograd.Function`; its backward run
inside the level reads it as live), whatever the grad mode, because their
kernels, the CUDA-graph replay and the custom operators carry no forward rule;
the reference runs instead and the gate records its usual decline reason. The
sea-ice gates do the same: the LSR line factorization and sweep and the two
ordered residual sums (`seaice_lsr_cuda` variants `factor` and `sweep`,
`seaice_ordered_sum_cuda` variants `ordered_sum` and `ordered_sum_pair`) take
their Python reference and record `inapplicable` with the operator's reason
(`non_cuda` and the other operand reasons first, `grad_enabled` on a CUDA
float64 call), the replayed LSR iterations (`seaice_lsr_graph`) record
`grad_enabled`, and an explicit request (the variable set by the user) raises
after recording, as for any inapplicable call. The reading is the program's
forward-mode state, not the tensors: an open level with no dual tensor in it
still declines every gate, giving up the kernels and the graph replay, makes
an explicit-ROOT CG3D solve take ROOT (so it must converge), and still raises
on the paths listed below. Close the level when no tangent is needed. Inside
the ROOT path both solves run in such a Function on plain tensors with forward
mode off, so those primal solves still use the graph replay and the fused
kernels when requested. The `fused` tendency assembly arm
(`DEFGCM_ASSEMBLY_ARM=fused`) sums through a compiled kernel; with forward
mode live it takes the eager chain instead, the same left-to-right sum, and
counts it in the arm's `fused-fallback` tally (`DEFGCM_ASSEMBLY` report).

These paths raise `NotImplementedError` instead of returning a value without
its tangent: `ModelRunner.run` with `track_gradients=False` (it detaches the
initial state and runs under inference mode), with a compile target, or with
more than one rank; a solve inside a `CommSession` or with peer ranks; the
communication operators; the checkpoint window (capture, run, and the reverse
entry, before admission and the objective's reverse; with a `CommSession` the
window then ends through its failure path, which notifies the other ranks, and
the caller receives `CommSessionError` with the `NotImplementedError` as its
cause); the implicit convective KPP; `checked_root_solve` (as `SolveRejected`,
eager VJP only); `abs_with_zero_slope` inside a compiled region (under
`fullgraph=False` the trace breaks there and the call runs eagerly instead). A
direct `pcg_solve(root_adjoint=True)` without `root_operands` raises only when
a tangent reaches the residual, because the coefficient tensors are needed for
`A_dot x`; without one it returns the solution as before. All of these read
program state, so the ranks of a job refuse together only when every rank
opens the same level; a rank with no dual tensor inside its open level refuses
as well.

With the nonlinear free surface (`nonlin_free_surf >= 1`) the surface hFac
keeps its dependence on `etaH` (`steppers/ifs/nlfs_geometry.py`,
`refresh_hfac_surf` no longer detaches it) and, with autograd on, is installed
as new tensors each step, so the dependence through the time-varying geometry
reaches both derivative modes. Forward-mode derivatives with a nonlinear free
surface have not been checked against central differences.

## Running your own case

### What a case file contains

| Section | Role |
|---|---|
| `run` | `experiment_name` (run directory name), `device` (`cpu`, `cuda`, `auto`), `start_date`, `precision` (`float64` default; `float32` accepted, *unverified*) |
| `grid` | `nx`, `ny`, `nz`, `center_lat`, `center_lon`, `resolution_deg`, `depth_m`, `vertical_layer_thicknesses` |
| `time` | `dt` and the per-equation steps; run length from `steps`, else `seconds`, else `days` |
| `physics`, `advection`, `boundary` | physical settings used by the MITgcm export and, in part, by model assembly |
| `mit_parallel`, `mit_comparison` | MITgcm tiling and switches; `mit_comparison.enabled` turns on stages 1c and 3 and the comparison in 4 and 5 (see "MITgcm comparison") |
| `sources` | named raw inputs: `type` and `path` |
| `pipelines` | stage 1a, 1b and 1c processing chains: source, operators, output consumer |
| `case` | `kind` and `extras` (initial-state, open-boundary and forcing sources, grid mode) |
| `model` | the LOCOMgcm modules: `dynamics` (ordered list of `name` and `kwargs`), `stepper`, `boundary_condition` |
| `cache`, `output` | prepared-input cache, run directory root, monitor interval and NetCDF cadence |

LOCOMgcm module parameters live in `model:`. The same physical quantity
can appear twice: in pureocean, `physics.viscosity` and the
`horizontal_viscosity` entry of `model.dynamics` both hold 500. The LOCOMgcm
run reads the `model.*` value; the `physics.*` value feeds the MITgcm export.
The pairs are listed in `defgcm/io/runner/config/_double_track.py`. An
overlay or `--set` item that writes one spelling updates the other, so one
override changes the parameter everywhere. A case file, or an override, that
writes the two spellings with different values stops stage 1a and stage 2
with `TranslationError: yaml double-tracks the following numerical
parameter(s) with inconsistent values across physics.* and model.*`.
Some `physics` fields do feed the model: `advection.vertical` sets the tracer
vertical scheme (from the code), and `physics.implicit_vertical_advection`
sets stepper options (verified by the build errors under Troubleshooting).
The nonhydrostatic pressure solve has no real-freshwater term (MITgcm's
`EmPmR*mass2rUnit`, `pre_cg3d.F:88-107`). With `use_real_freshwater_flux: true`,
every step checks the EmPmR it will use and raises `NotImplementedError` before
the step runs when that EmPmR is nonzero on any wet cell, requires grad or
carries a forward-mode tangent; turn the real freshwater flux off or run
hydrostatic. A constant zero or absent EmPmR runs as before. The check covers the
volume flux of a `wpflux_surface_forcing` module, the only freshwater input this
term would read; the ERA5 bulk and sea-ice freshwater fluxes do not reach it, and
with them alone the free-surface solve raises for the missing EmPmR. On several
ranks each rank checks its own tile.

### `--overlay` and `--set`

Layers merge in this order, later ones winning: built-in defaults, the case
file, each `--overlay` file in command-line order, then each `--set key=value`.
Nested mappings merge key by key. Lists are replaced as a whole.

`examples/pureocean_cpu.yaml` is an overlay:

```yaml
run:
  device: cpu
  experiment_name: pureocean_cpu
time:
  steps: 60
output:
  snapshot_interval_seconds: 2400
  nc_output_interval_seconds: 2400
sources:
  hycom: {path: 'external:synthetic_pureocean_dir/hycom'}
  era5: {path: 'external:synthetic_pureocean_dir/era5'}
  gebco: {path: 'external:synthetic_pureocean_dir/gebco.nc'}
```

`--set` overrides one field. Adding `--set time.steps=20` to the quick-start
command ran 20 steps, so `--set` won over the overlay's 60. Other useful
items:

- `--set run.device=cpu`
- `--set output.snapshot_interval_seconds=600 --set output.nc_output_interval_seconds=600`
  writes a frame every 5 steps at dt = 120 s. The `_seconds` fields win over
  the `_hours` fields.
- `--set sources.hycom.path=/abs/path/to/hycom`

`model.dynamics` is a list of named entries. Address one entry by name, in
`--set` and in overlay files alike:

```sh
--set 'model.dynamics[horizontal_viscosity].kwargs.viscosity=300'
```

```yaml
model:
  dynamics[horizontal_viscosity]:
    kwargs: {viscosity: 300}
```

The entry keeps its other settings and its place in the list. Writing a
mapping over the list (`--set model.dynamics.9.kwargs.viscosity=300`) or
naming an entry that does not exist stops the run before any stage starts
with one line, `ERROR configuration does not load: ...`, that lists the known
entry names (exit code 5; the previous outputs and manifest are untouched).
Supplying the whole list in an overlay still
replaces it; entries such as `coriolis_parameter: $coriolis_parameter` are
placeholders that the model build resolves from the grid, copy them unchanged.

### Changing a parameter and rerunning

Prepared inputs do not depend on model parameters, so a sweep prepares once
and reruns stage 2. Stage 2 finds its inputs through `manifest.yaml` in the
run directory, which is named by `run.experiment_name`. Keep that name, pass
the new overlay, and select stage 2 only:

```sh
python -m experiments.pipeline_v2.run_pipeline \
  --config experiments/configs/cases/pureocean.yaml \
  --overlay examples/pureocean_cpu.yaml \
  --set 'model.dynamics[horizontal_viscosity].kwargs.viscosity=100' \
  --run-id visc100 --stages 2
```

The one `--set` item also updates `physics.viscosity`, because the two are a
double-tracked pair. Stage 2 reran (no `--force` needed) and the last frame's
max |u| moved from 3.0553e-02 to 3.0650e-02 m/s. The output overwrites
`nc_snapshots/`, so copy it elsewhere between variants. If an overlay changes
`run.experiment_name`, stage 2 looks in a new empty directory and stops with
`PREREQUISITE CHECK FAILED ... stage1b_export_to: not present in manifest`;
then stages 1a and 1b must run again (about 8 s for pureocean).

Stage 2 applies the same `physics.*` versus `model.*` consistency check as
stage 1a; a case or override that writes the two spellings differently stops
the rerun with the `TranslationError` (exit code 5, stage marked FAILED in the
manifest, output files untouched). The record of a stage-2-only rerun is
`_pipeline_meta/stage2_torch_run/stage2_config.yaml` (run id, case file,
overlays as typed, `--set` items and the merged configuration).
`metadata.yaml` is rewritten too; its `model_args` take the stepper kwargs
first and `case.extras` only as a fallback. `config_snapshot/`, and `run_id`
in `manifest.yaml`, keep the values from stage 1a.

A run shorter than the case horizon still prepares inputs for `time.days`.
Stage 1a's time window comes from `time.days`, not from `time.steps`. For
pureocean, that means 9 daily HYCOM files from `start_date` (the reader takes
`ceil(days + 1) + 1`) and ERA5 covering at least 7 days. When `time.steps` is
set, `--set time.days=...` is ignored with a warning (from the code); use
`time.steps` or `time.seconds` to change the run length.

### MITgcm comparison (optional)

LOCOMgcm runs without MITgcm; `mit_comparison.enabled` turns the MITgcm
path on. With it true, stage 1a translates the configuration into MITgcm
parameters, rejects any yaml key the MITgcm parameter table does not claim
(`TranslationError: yaml key '...' maps to no legacy arg`), and writes the
parameters to `_pipeline_meta/model_input/run_params.yaml` for stage 1c.
With it false, stage 1a skips that translation but still runs the
double-track check; `run_params.yaml` then holds only
`experiment_name`, `device`, `resolution_deg`, `fixed_dt`, `steps` and
`snapshot_interval_seconds`, and `metadata.yaml` records
`type: defgcm_only` without a reference source. The switch does not
change the LOCOMgcm result: `day001.nc` was byte-identical either way. To
add the comparison to pureocean, rerun from stage 1a with
`--set mit_comparison.enabled=true` and a configured MITgcm tree. With
`run_case.sh`, `MIT=1` or `MIT=0` passes the same setting to every stage and
is replayed on `resume`.

### Input files

`sources.<name>.path` names an external key, such as `external:hycom_dir`.
Set its absolute location in `DEFGCM_PATHS` or the corresponding environment
variable; see [input configuration](INPUTS.md#case-data-locations). Absolute
path overrides also work. Relative raw-input paths are rejected. The readers
expect these layouts (the synthetic generator writes them):

| Source type | Expected files |
|---|---|
| `hycom_netcdf` | daily `hycom_uvts_YYYYMMDD.nc` under the path (searched recursively), variables `water_temp`, `salinity`, `water_u`, `water_v`, `surf_el` on `time`, `depth`, `lat`, `lon` |
| `era5_netcdf` | `<path>/<year>/era5_u10_<year>.nc` and `era5_v10_<year>.nc`. Stage 2 opens this layout directly. Stage 1a accepts any `era5_*.nc` under the path. |
| `gebco_netcdf` | one file with `elevation(lat, lon)`, negative below sea level |

Data sources, acquisition, the prepared-input archives for the published
experiments and MITgcm paths are described in [INPUTS.md](INPUTS.md). The
archives restore prepared run directories so that later stages can be run
alone (*unverified* here).

## GPU and native kernels

### Device

`run.device` is `cpu`, `cuda` or `auto`. `auto` picks CUDA when
`torch.cuda.is_available()`, otherwise CPU (verified with
`CUDA_VISIBLE_DEVICES=` set to empty). With library defaults, pureocean stage 2
on an RTX 4060 (`--set run.device=cuda`) compiled nothing and printed the same
monitor values as the CPU run; over 60 steps every variable agreed with the
CPU run to within 6.2e-14.

`DEFGCM_COMPILE_TARGET` selects `torch.compile` in stage 2: `disabled`,
`tendency` (the single-process default) or `model_step`. For pureocean,
`tendency` compiles nothing: the compile record shows `unique_graphs: 0`,
because this stepper does not call the wrapped tendency methods.
`model_step` was not tested.

### Switches

Kernels are selected by environment variables read when the program starts.
Importing `defgcm` while a retired `TO_*` or `TORCHOCEAN_*` name is set raises
`RuntimeError` and lists each one with its `DEFGCM_*` replacement; the table is
`defgcm/_legacy_env.py`.
Booleans accept `1/true/yes/on` and `0/false/no/off`. All native CUDA kernels
are off by default. At exit, every process that imported `defgcm` prints
one `DEFGCM_ACCEL {...}` JSON line per switch to stderr (`DEFGCM_ACCEL_REPORT=0`
turns this off). Its `ignition_state` is `disabled`, `requested-zero` (enabled
but never reached), `requested-hit` (ran) or `inapplicable-or-error`, with
reasons in `counts`. A correct result does not show that a kernel ran; check
these lines. In a pipeline run they appear on the stderr of the
`run_pipeline` process; `stage2_stdout.log` only carries the
`DEFGCM_ACCEL_PRELOAD` lines, which `DEFGCM_ACCEL_REPORT=0` does not
suppress. With `DEFGCM_EOS_CUDA=1` over 60 steps the line
for that switch read `"ignition_state":"requested-hit"` with 66 calls at
`JMD95EquationOfState.density_fields`.

Purposes below are summarized from `defgcm/accel/registry.py`. "CPU
verified" means observed in a pureocean CPU run, which produced the same
diagnostics as a run without the switch.

| Variable | Default | Purpose | On a CPU device |
|---|---|---|---|
| `DEFGCM_CG2D_FUSED_PCG` | 0 | fused CUDA kernels for the 2-D free-surface PCG iteration | declined (`non_cuda`), CPU verified |
| `DEFGCM_CG2D_CUDA_GRAPH` | 0 | replay the 2-D PCG solve as a captured CUDA graph | declined (`non_cuda`), CPU verified |
| `DEFGCM_CG3D_FUSED_PCG` | 0 | fused CUDA kernels for the nonhydrostatic 3-D PCG | not exercised (hydrostatic case) |
| `DEFGCM_CG3D_USE_FAST_THOMAS` | 1 | CUDA vertical Thomas preconditioner in CG3D | not exercised (hydrostatic case) |
| `DEFGCM_IVADV_FUSED_BACKEND` | unset (`cuda` only) | fused CUDA implicit vertical advection solve | raises `native fused vertical CUDA dispatch requires a CUDA float64 or float32 rhs` when implicit vertical advection is on; no effect otherwise (verified) |
| `DEFGCM_TRIDIAG_CUDA` | 0 | CUDA tridiagonal solve | declined (`non_cuda`), CPU verified |
| `DEFGCM_TRIDIAG_BUILD_CUDA` | 0 | CUDA tridiagonal system build | not tested |
| `DEFGCM_EOS_CUDA` | 0 | nonlinear equation of state (JMD95, MDJWF) | declined (`non_cuda`), CPU verified; ran on GPU (`requested-hit`, 23 calls in 20 steps) |
| `DEFGCM_VIM_FUSED_CUDA` | 0 | fused horizontal vector-invariant momentum phase | declined (`non_cuda`), CPU verified |
| `DEFGCM_VIM_CORIOLIS_U_CUDA` | 0 | U-face Coriolis and vorticity-flux stencil | not tested |
| `DEFGCM_HVISC_FUSED_CUDA` | 0 | fused Leith/Smagorinsky viscosity | no consumer with constant viscosity (`requested-zero`) |
| `DEFGCM_DST3_CUDA` | 0 | DST3 advection fluxes | not exercised (upwind case) |
| `DEFGCM_TRACER_ADV_FUSED_CUDA` | 0 | fused horizontal DST3 tracer advection | not exercised (upwind case) |
| `DEFGCM_KPP_STATEKPP_CUDA`, `DEFGCM_KPP_BLDEPTH_CUDA`, `DEFGCM_KPP_BLMIX_CUDA` | 0 | KPP sub-steps on CUDA | not exercised (no KPP) |
| `DEFGCM_KPP_FUSED_CUDA` | 0 | whole regional KPP phase in one CUDA op | not exercised (no KPP) |
| `DEFGCM_CACHE_CONSTANTS`, `DEFGCM_DEVICE_RESIDENT_MASKS`, `DEFGCM_IMPLDIFF_BUILD_BATCH`, `DEFGCM_VECTORIZE_VERTSHEAR`, `DEFGCM_KPP_WSCALE_BATCH`, `DEFGCM_KPP_STATIC_HALO_CACHE`, `DEFGCM_KPP_FACE_BATCH`, `DEFGCM_CUBE_LOCAL_GRID_CACHE` | 1 | caches and batched PyTorch paths, no native build | the first four ran on CPU (`requested-hit`) |
| `DEFGCM_COMPILE_FULLGRAPH`, `DEFGCM_COMPILE_DYNAMIC`, `DEFGCM_COMPILE_PIN_SCHEDULE` | 0, 0, 1 | `torch.compile` options (one graph per step, which raises before the first step for reverse mode on more than one rank without a checkpoint window; dynamic shapes, a single-rank forward diagnostic that raises before the first step with gradients or on more than one rank; multi-rank scheduling) | not tested |

`DEFGCM_CHECK_LEVEL` (`production` default, `debug`, `strict`) sets runtime
checks. `DEFGCM_DEFERRED_CHECKS` is retired.

The SLURM driver `experiments/pipeline_v2/run_case.sh` adds shorthand
variables that the Python pipeline does not read. Read from the script, not
run:

| run_case.sh variable | Default | Sets |
|---|---|---|
| `FUSED` | 1 | `DEFGCM_CG3D_FUSED_PCG=1` |
| `IVADV` | 1 | `DEFGCM_IVADV_FUSED_BACKEND=cuda`, only when `run.device` is `cuda` |
| `STATEKPP` (`TRITON_KPP` is a deprecated alias) | 1 | `DEFGCM_KPP_STATEKPP_CUDA=1`; the kernel runs only with `stepper.kpp_dbloc_form: divided_difference` ([numerics](NUMERICS.md)) |
| `DEFCHK` | 1 | `DEFGCM_DEFERRED_CHECKS` (retired), only on `cuda` |
| `N2FUSED` | 0 | `DEFGCM_KPP_FUSED_CUDA=1 DEFGCM_VIM_FUSED_CUDA=1 DEFGCM_TRACER_ADV_FUSED_CUDA=1` |

### Where native kernels compile

Native kernels are built on first use with `torch.utils.cpp_extension.load`.
This needs the CUDA toolkit, a host C++ compiler and `ninja`. torch 2.9 looks
for the toolkit in this order: `CUDA_HOME`, `CUDA_PATH`, an `nvcc` on `PATH`,
then `/usr/local/cuda`. For the host compiler, the kernel loaders always
look next to the Python interpreter and on `PATH` for
`x86_64-conda-linux-gnu-g++`, `g++`, then `c++`, and pass the result to
`nvcc` as `-ccbin=`. `CXX`, when already set (conda environments set it),
is kept and compiles the `.cpp` files; when `CC` is set, torch adds a second
`-ccbin $CC`. Verified combination: torch 2.9.0+cu128 with the CUDA 13.0
`nvcc` from `/usr/local/cuda` and the conda `x86_64-conda-linux-gnu-g++`
and `-c++` built `defgcm_eos_cuda` in 33 s without a version warning.
Builds go to
`~/.cache/torch_extensions/py<python>_cu<cuda>/defgcm_<kernel>/`
(for example `py311_cu128/defgcm_eos_cuda/`). When `TORCH_EXTENSIONS_DIR`
is set, they go to `$TORCH_EXTENSIONS_DIR/defgcm_<kernel>/`. The first
build added about 34 s to stage 2; the next process reused it (stage 2
runtime 2.2 s for 20 steps, 4.3 s for 60). After a successful early load, the stage 2
log shows `DEFGCM_ACCEL_PRELOAD key=<kernel> status=loaded`. Setting
`DEFGCM_<KERNEL>_VERBOSE=1` (for example `DEFGCM_EOS_CUDA_VERBOSE=1`) passes
`verbose=True` to the build (*unverified*).

### Stuck lock

Symptom: a GPU run prints the first step line, or stops after
`[3/4] Running LOCOMgcm ...`, then sits with near-zero CPU and no further
output. This happens after an earlier build was killed, for example by
`scancel` or SIGKILL.

Cause: each build directory holds a `lock` file while a process builds.
torch creates it, and another process that finds it polls in
`torch/utils/file_baton.py` until it disappears. torch 2.9 sets no timeout and
prints no warning. A killed build never removes its lock. Every later
`load()` of that kernel waits forever, even when the compiled `.so` is
already present. The wait was verified with a planted lock in an empty
`TORCH_EXTENSIONS_DIR`: `load()` made no progress for 15 s and printed
nothing. From the code: kernels with an early-load hook (`cg2d_fused_pcg`,
`cg3d_fused_pcg`, `ivadv`, `kpp_fused_cuda`, `eos_cuda`, `vim_fused_cuda`,
`tracer_adv_fused_cuda`, `hvisc_fused_cuda`) block before the `step 0` line;
the others block at first use, after it.

Cure, on the machine that ran the build:

```sh
CUDA_VISIBLE_DEVICES=0 bin/jit_lock_sweep.sh [extra_dir ...]
```

The script searches `TORCH_EXTENSIONS_DIR` (default
`~/.cache/torch_extensions`) and every directory given as an argument for
`lock` files. It removes those that no live process holds, using `fuser`, and
leaves held ones in place. Without `CUDA_VISIBLE_DEVICES`, it exits silently
without doing anything, which is usually the case on a workstation. With it
set, the planted lock was removed:
`[jit-lock] removing stale build lock (...): .../defgcm_eos_cuda/lock`.
Manual alternative: find the files with
`find ~/.cache/torch_extensions -name lock`. Confirm that `fuser <file>`
prints nothing and that no build of yours is running, then delete the file.

## Troubleshooting

| Message | Cause and fix |
|---|---|
| `ExternalPathError: missing external path key ...` | Configure the named key as described in [INPUTS.md](INPUTS.md), or generate the synthetic inputs and set `DEFGCM_SYNTHETIC_PUREOCEAN_DIR` before using `--overlay examples/pureocean_cpu.yaml`. |
| Stage 2 exits 5; its log says `[framework] failed: No CUDA GPUs are available` | The case file sets `run.device: cuda`. Use `--set run.device=cpu` or `auto`. |
| `[framework] failed: native fused vertical CUDA dispatch requires a CUDA float64 or float32 rhs` | `DEFGCM_IVADV_FUSED_BACKEND=cuda` on a CPU run with implicit vertical advection. Unset it. |
| `build_model_from_manifest: ... enable_vertical_shear=true conflicts with momentum_implicit_vertical_advection=true` (plus a bottom-viscosity requirement) | Momentum implicit vertical advection is incompatible with pureocean's vertical-shear setting. Tracer-only `physics.implicit_vertical_advection.{temperature,salinity}=true` builds. |
| `RuntimeError: helper.preprocess requires mitgcm/run/ ... (stage3 not yet run)` | Stage 4 with `mit_comparison.enabled: true` needs MITgcm output from stage 3. Set it false for a LOCOMgcm-only run. |
| `the MITgcm stages [...] need mit_comparison.enabled=true` (before any stage) or `stage1c_export_mit failed: the MITgcm export needs mit_comparison.enabled=true` | Stage 1c, 3 or 5 was requested (for example by `--stages all`) with the comparison off. Rerun with `--set mit_comparison.enabled=true`, or leave those stages out. |
| `AttributeError: module 'torch.compiler' has no attribute 'nested_compile_region'` at import | torch too old (seen with 2.5.1). Use 2.9.0. |
| Stage 3: `MITgcm did not exit normally (exit=255)`, `mpirun.err` shows `unrecognized argument oversubscribe` | The default launcher flags are for OpenMPI. `MITGCM_MPI_OVERSUBSCRIBE=0` drops the flag (*unverified*). |
| A failed stage prints a Python traceback and `manifest.yaml` keeps `overall_status: RUNNING` | Some stage errors escape the orchestrator. Read the traceback, fix the input and rerun the stage. |
| GPU run idle after the first step line | Stale build lock; see "Stuck lock". |
| A second `ModelRunner.run` on the same model gives different numbers | The model keeps step history. Build a new model per run. |
| `NotImplementedError: a Flather barotropic open boundary with nonlin_free_surf=... is not supported` at the first step | A Flather velocity boundary (alone or inside `tracer_open`) with the nonlinear free surface. The step freezes the Flather target with H read from the lagged surface geometry, and its last application averages on the new one; no discretisation of this pairing is defined. Use `nonlin_free_surf: 0`, or the `prescribed_3d` velocity boundary. |
| `ValueError: Flather balance_factors {...}: a negative factor (separate per-side balance) is not implemented` | MITgcm balances a side with `OBCS_balanceFac < 0` on its own; the Flather boundary implements the joint balance only, shared with `prescribed_3d` and the momentum sponge. Use factors of 0 or more (0 counts the side's inflow without correcting it). |
| `ValueError: DEFGCM_COMPILE_DYNAMIC=1 (RunnerCompileConfig.dynamic=True) is not supported for a differentiated run` | Dynamic-shape compilation is a forward-only diagnostic; it is refused before anything compiles at `ModelRunner.run` with gradients, at a checkpoint window's entry and in the compile context. Unset `DEFGCM_COMPILE_DYNAMIC` (static shapes), or run without gradients. |
| `ValueError: DEFGCM_COMPILE_FULLGRAPH=1 (RunnerCompileConfig.fullgraph=True) is not supported for reverse mode on more than one rank without a checkpoint window` | Without a window the communication session runs each halo exchange eagerly at a graph exit, which one graph per step cannot trace. Differentiate through a checkpoint window, or set `DEFGCM_COMPILE_FULLGRAPH=0`. |
| `NotImplementedError: nonhydrostatic pressure solve with use_real_freshwater_flux=True and ...` | This step's EmPmR is nonzero on a wet cell, requires grad or carries a forward-mode tangent, and the nonhydrostatic solve has no real-freshwater term; see the end of "What a case file contains". Turn the real freshwater flux off or run hydrostatic. |
| `ValueError: nonhydrostatic pressure solve: mass2rUnit = 1/rhoConst needs rhoConst ...` | The nonhydrostatic solve takes rhoConst from the `hydrostatic_pressure_gradient` module's `reference_density` (MITgcm `mass2rUnit = 1/rhoConst`). Add that module; no default density is assumed. |
| `TranslationError: yaml double-tracks the following numerical parameter(s) with inconsistent values across physics.* and model.*` | A quantity written in both `physics.*` and `model.*` has two different values. Give both the same value, or write only one of them in an overlay or `--set` item (the other follows); see "What a case file contains". Stages 1a and 2 both check. |
| `configuration does not load: 'dynamics' is a list of named entries; address one as 'dynamics[<name>]' ...` | `--set` or an overlay wrote a mapping over the `model.dynamics` list. Use `model.dynamics[<name>].kwargs.<key>`. |
| `configuration does not load: 'dynamics' has no entry named '...'; known names: [...]` | The entry name in `model.dynamics[<name>]` is misspelled. Pick one of the listed names. |
| `PREREQUISITE CHECK FAILED for stage2_torch_run ... stage1b_export_to: not present in manifest` | Stage 2 was run alone in a run directory that has no prepared inputs, usually because `run.experiment_name` changed. Keep the name or rerun stages 1a and 1b. |
| `UnstableSpecificationWarning: The data type (FixedLengthUTF32...)` during stage 1a | Warning from zarr when writing a string coordinate. The stage completes. |
| 28 `DEFGCM_ACCEL {...}` lines on stderr at exit | Normal accelerator report; see "Switches". `DEFGCM_ACCEL_REPORT=0` turns it off. |
