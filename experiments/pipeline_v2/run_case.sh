#!/usr/bin/env bash
# Run a configured LOCOMgcm and MITgcm case through SLURM.
#
# Usage:
#   run_case.sh <config.yaml> <run-id>
#   run_case.sh status <run-id>
#   run_case.sh resume <run-id>
#
# STAGES selects 1a (prepare), 1b (LOCOMgcm inputs), 1c (MITgcm inputs),
# 2 (LOCOMgcm), 3 (MITgcm), 4 (postprocessing), and 5 (comparison).
# Without STAGES, MIT=1 runs every stage and MIT=0 runs 1a,1b,2,4, as
# run_pipeline.py does; MIT defaults to yaml mit_comparison.enabled.
# STEPS and SECS override run length; otherwise time.steps, time.seconds and
# time.days are used in that order. SNAPSHOT_SECONDS or SNAPSHOT_HOURS
# overrides output cadence.
#
# LOCOMGCM_PYTHON selects Python; the default is the active interpreter. Worker
# processes inherit the environment. LOCOMGCM_ENV_SETUP may supply shell
# commands to initialize the worker environment.
#
# CPU_PARTITION/CPU_NODELIST select preprocessing and postprocessing placement.
# LOCOMGCM_PARTITION/LOCOMGCM_NODELIST select LOCOMgcm placement. MIT_PARTITION and
# MIT_NODELIST default to the CPU selections. Empty values let SLURM choose.
# LOCOMGCM_NPROC=1 and LOCOMGCM_NNODES=1 select GPU process and node counts; LOCOMGCM_GRES defaults
# to gpu:$LOCOMGCM_NPROC. LOCOMGCM_NPX/LOCOMGCM_NPY set the layout.
# Multi-node runs accept LOCOMGCM_RDZV_ENDPOINT, LOCOMGCM_MASTER_PORT and LOCOMGCM_RDZV_ID.
# LOCOMGCM_NODE_RANK defaults to SLURM_NODEID. LOCOMGCM_DRY_RUN=1 prints the GPU launch.
#
# PREP_MEM/LOCOMGCM_MEM/MIT_MEM use grid-dependent defaults; POST_MEM defaults to 192G.
# PREP_CPUS=16, LOCOMGCM_CPUS=2, POST_CPUS=16 and MIT_RANKS select CPU resources.
# PREP_MIN=120, LOCOMGCM_HOURS=4, MIT_HOURS=4 and POST_HOURS=2 select time limits.
# MITGCM_ROOT, MITGCM_OPTFILE and MITGCM_GENMAKE_FLAGS configure the native build.
# MITGCM_MPI_LAUNCHER defaults to mpirun. MIT_MPI_ENV_FILE may name a shell
# environment file to source before native execution.
#
# IVADV, STATEKPP, FUSED and DEFCHK default to 1; N2FUSED defaults to 0.
# They control implicit vertical CUDA solves, CUDA KPP, fused CG3D solves,
# deferred device checks and phase-fused CUDA operators, respectively.
# Set individual switches to 0 to select their reference execution paths.
#
# CLEAN=1 removes a previous run. SKIP_PREP=1 with REUSE_PREP_FROM reuses inputs;
# SKIP_MIT=1 with REUSE_MIT_FROM reuses native output. RESUME_CKPT selects a
# LOCOMgcm checkpoint. CACHE, CACHE_VERIFY and CACHE_REBUILD control prepared
# input reuse. METRICS_FULL=1 enables the full diagnostic set.
# Logs are written to outputs/logs/<run-id>/ and results to outputs/data/.
# Inspect the comparison and diagnostic results after the stages complete.

set -euo pipefail

cd "${LOCOMGCM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ROOT=$(pwd)
# LOCOMGCM_RUN_LOG_ROOT selects the transient logs; durable invocation metadata stays with the run.
RUN_LOG_ROOT=${LOCOMGCM_RUN_LOG_ROOT:-$ROOT/outputs/logs}
PYBIN=${LOCOMGCM_PYTHON:-$(command -v python)}
PYBIN=$(command -v "$PYBIN")
export LOCOMGCM_REPO="$ROOT"

# --- subcommands -----------------------------------------------------------
# `status <run-id>`  → print the run's pipeline state and exit (no SLURM).
# `resume <run-id>`  → re-run derived ENTIRELY from the run's own records:
#   config from config_snapshot/source_config.yaml, env overrides (STAGES /
#   SNAPSHOT_SECONDS / METRICS_FULL / ...) replayed from resolved_invocation.yaml.
#   So neither the operator nor a fresh session re-specifies the yaml or the
#   dump interval. A command-line env (e.g. `STAGES=4,5 ... resume <id>`) WINS
#   over the replayed value (the ${K:-...} default form below).
case "${1:-}" in
--help|-h)
    sed -n '2,/^set -euo pipefail/{ /^#/ { s/^# *//; p; }; }' "${BASH_SOURCE[0]}"
    exit 0
    ;;
status)
    [ -z "${2:-}" ] && { echo "usage: $0 status <run-id>" >&2; exit 2; }
    exec "$PYBIN" -m experiments.pipeline_v2.show_run_status "$2"
    ;;
resume)
    [ -z "${2:-}" ] && { echo "usage: $0 resume <run-id>" >&2; exit 2; }
    _RID=$2
    CONFIG=$ROOT/outputs/data/$_RID/config_snapshot/source_config.yaml
    [ -f "$CONFIG" ] || { echo "resume: no config_snapshot/source_config.yaml for $_RID" >&2; exit 2; }
    _INVO_DURABLE=$ROOT/outputs/data/$_RID/_pipeline_meta/resolved_invocation.yaml
    _INVO_LOG=$RUN_LOG_ROOT/$_RID/resolved_invocation.yaml
    if [ -f "$_INVO_DURABLE" ]; then
        _INVO=$_INVO_DURABLE
    elif [ -f "$_INVO_LOG" ]; then
        _INVO=$_INVO_LOG
    else
        _INVO=""
    fi
    if [ -n "$_INVO" ]; then
        eval "$("$PYBIN" - "$_INVO" <<'PY'
import sys, shlex, yaml
env = (yaml.safe_load(open(sys.argv[1])) or {}).get("env", {}) or {}
for k in (
    "STAGES",
    "MIT",
    "STEPS",
    "SECS",
    "SNAPSHOT_SECONDS",
    "SNAPSHOT_HOURS",
    "METRICS_FULL",
    "CACHE",
    "FUSED",
    "IVADV",
    "TRITON_KPP",
    "STATEKPP",
    "DEFCHK",
    "N2FUSED",
    "MITGCM_ROOT",
    "MITGCM_OPTFILE",
    "MITGCM_MPI_LAUNCHER",
    "MIT_MPI_ENV_FILE",
    "LOCOMGCM_COMPILE_TARGET",
    "LOCOMGCM_PARTITION",
    "LOCOMGCM_NODELIST",
    "LOCOMGCM_NPROC",
    "LOCOMGCM_NNODES",
    "LOCOMGCM_GRES",
    "LOCOMGCM_CPUS",
    "LOCOMGCM_MEM",
    "LOCOMGCM_HOURS",
    "LOCOMGCM_LAUNCH_PREFIX",
    "LOCOMGCM_RDZV_ENDPOINT",
    "LOCOMGCM_MASTER_PORT",
    "LOCOMGCM_RDZV_ID",
    "LOCOMGCM_NPX",
    "LOCOMGCM_NPY",
    "LOCOMGCM_NODE_RANK",
    "PREP_MEM",
    "PREP_CPUS",
    "PREP_MIN",
    "MIT_MEM",
    "MIT_RANKS",
    "MIT_HOURS",
    "MIT_PARTITION",
    "MIT_NODELIST",
    "CPU_PARTITION",
    "CPU_NODELIST",
    "POST_HOURS",
    "POST_MEM",
    "POST_CPUS",
):
    v = env.get(k)
    if v not in (None, ""):
        print(f'export {k}="${{{k}:-{shlex.quote(str(v))}}}"')
PY
)"
        echo "[resume] config=$CONFIG  (overrides replayed from $_INVO; command-line env wins)"
    else
        echo "[resume] WARNING: no resolved_invocation.yaml found for $_RID — env knobs will use schema defaults" >&2
    fi
    # Resume = continue from the newest checkpoint, NOT restart from step 0.
    # Without this the LOCOMgcm stage runs with checkpoint.resume=none (schema default)
    # and silently re-runs from the IC — config_snapshot has no resume key.
    # Override with RESUME_CKPT=<step> (or =none to force a fresh step-0 re-run).
    export RESUME_CKPT="${RESUME_CKPT:-latest}"
    echo "[resume] checkpoint.resume=$RESUME_CKPT (load newest checkpoint; set RESUME_CKPT=none for a step-0 re-run)"
    set -- "$CONFIG" "$_RID"
    ;;
esac

if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
    echo "usage: $0 <config.yaml> <run-id>   |   $0 status <run-id>   |   $0 resume <run-id>" >&2
    exit 2
fi
CONFIG=$1
RUN_ID=$2

if [ ! -f "$CONFIG" ]; then
    echo "config not found: $CONFIG" >&2
    exit 2
fi

LOG_DIR=$RUN_LOG_ROOT/${RUN_ID}
mkdir -p "$LOG_DIR"
MASTER_LOG=$LOG_DIR/master.log
PREP_LOG=$LOG_DIR/phase0_prep.log
LOCOMGCM_LOG=$LOG_DIR/phase1_TO.log
MIT_LOG=$LOG_DIR/phase1_MIT.log
POST_LOG=$LOG_DIR/phase2_postproc.log

log() { echo "[$(date +%T)] $*" | tee -a "$MASTER_LOG"; }

# ---------------------------------------------------------------------
# Auto-derive everything possible from the yaml (single python read).
# Prints `KEY=VALUE` shell assignments on stdout; validation failures go
# to stderr + nonzero exit so the `|| exit` below trips.
# ---------------------------------------------------------------------
DERIVED=$(CONFIG="$CONFIG" STEPS="${STEPS:-}" SECONDS_OVR="${SECS:-}" "$PYBIN" <<'PY'
import os, sys, yaml

cfg = yaml.safe_load(open(os.environ["CONFIG"])) or {}
g  = cfg.get("grid", {}) or {}
mp = cfg.get("mit_parallel", {}) or {}
t  = cfg.get("time", {}) or {}
run = cfg.get("run", {}) or {}
mc = cfg.get("mit_comparison", {}) or {}

def need(d, k, where):
    if k not in d or d[k] is None:
        print(f"[run_case] yaml missing {where}.{k}", file=sys.stderr); sys.exit(1)
    return d[k]

nx = int(need(g, "nx", "grid")); ny = int(need(g, "ny", "grid")); nz = int(need(g, "nz", "grid"))
nPx = int(mp.get("nPx", 1)); nPy = int(mp.get("nPy", 1))
dt = float(need(t, "dt", "time"))
exp = str(run.get("experiment_name", "default"))
mit_enabled = bool(mc.get("enabled", True))
ranks = nPx * nPy

# --- detection only (enforcement is in bash, gated on whether MIT runs) ---
# grid divisibility (MIT tiling): nx%nPx, ny%nPy
div_msgs = []
if nx % nPx != 0: div_msgs.append(f"grid.nx {nx} not divisible by nPx {nPx}")
if ny % nPy != 0: div_msgs.append(f"grid.ny {ny} not divisible by nPy {nPy}")
div_ok = 0 if div_msgs else 1
div_msg = "; ".join(div_msgs)

# stage1c compute_mpi_topology target_nProcs vs nPx*nPy (MIT SIZE.h coherence)
s1c = (cfg.get("pipelines", {}) or {}).get("stage1c_export_mit", []) or []
mismatches = []
for pipe in s1c:
    for op in (pipe.get("operators", []) or []):
        if op.get("op") == "compute_mpi_topology":
            tgt = op.get("target_nProcs")
            if tgt is not None and int(tgt) != ranks:
                mismatches.append(f"{pipe.get('name')}={tgt}")
topo_ok = 0 if mismatches else 1
topo_msg = (f"stage1c target_nProcs ({', '.join(mismatches)}) != nPx*nPy={ranks} "
            f"(MIT SIZE.h topology would not match mpirun -np {ranks}); "
            f"fix yaml mit_parallel and stage1c target_nProcs together") if mismatches else ""

# --- run length (display + override resolution) ---
def _steps_from_yaml():
    if t.get("steps") is not None:
        return int(t["steps"]), "yaml time.steps"
    if t.get("seconds") is not None:
        s = t["seconds"]
        if isinstance(s, str):
            s = eval(s, {"__builtins__": {}}, {})  # author-controlled arithmetic
        return round(float(s) / dt), "yaml time.seconds"
    if t.get("days") is not None:
        return round(float(t["days"]) * 86400.0 / dt), "yaml time.days"
    return None, "unset"

steps_ovr = os.environ.get("STEPS", "").strip()
sec_ovr = os.environ.get("SECONDS_OVR", "").strip()
if steps_ovr:
    n_steps, src = int(steps_ovr), "override STEPS"
    time_set = f"--set time.steps={int(steps_ovr)}"
elif sec_ovr:
    n_steps = round(float(sec_ovr) / dt)
    src = f"override SECONDS={sec_ovr}"
    time_set = f"--set time.steps={n_steps}"
else:
    n_steps, src = _steps_from_yaml()
    time_set = ""  # let the yaml decide everywhere

cells_m = nx * ny * nz / 1e6
if cells_m < 5:     prep, to, mit = 128, 32, 64
elif cells_m < 30:  prep, to, mit = 256, 48, 128
elif cells_m < 80:  prep, to, mit = 512, 64, 192
else:               prep, to, mit = 768, 96, 256

def emit(k, v): print(f"{k}={v}")
emit("D_NX", nx); emit("D_NY", ny); emit("D_NZ", nz)
emit("D_NPX", nPx); emit("D_NPY", nPy); emit("D_RANKS", ranks)
emit("D_DT", dt); emit("D_EXP", repr(exp))
emit("D_MIT_ENABLED", 1 if mit_enabled else 0)
emit("D_NSTEPS", n_steps if n_steps is not None else "?")
emit("D_NSTEPS_SRC", repr(src).replace(" ", "_"))
emit("D_TIME_SET", repr(time_set))
emit("D_MEM_PREP", f"{prep}G"); emit("D_MEM_TO", f"{to}G"); emit("D_MEM_MIT", f"{mit}G")
emit("D_CELLS_M", round(cells_m, 1))
emit("D_DIV_OK", div_ok); emit("D_DIV_MSG", repr(div_msg))
emit("D_TOPO_OK", topo_ok); emit("D_TOPO_MSG", repr(topo_msg))
emit("D_CACHE_ROOT", repr((cfg.get("cache", {}) or {}).get("root", "outputs/data/_cache/")))
# run.device decides which accelerated kernels may be injected: the fused ivadv
# backends are CUDA-fp64-only and RAISE (deliberately, no silent fallback) on
# CPU/fp32/grad inputs, so a cpu-device case must not get them.  Default cuda
# matches the schema default used by the LOCOMgcm stage.
emit("D_DEVICE", repr(str(run.get("device", "cuda"))))
PY
) || { echo "[run_case] yaml derivation failed (see above)"; exit 1; }
eval "$DERIVED"
# D_TIME_SET / D_NSTEPS_SRC are repr()-quoted on the python side so eval parses
# them as single tokens (preserving spaces / emptiness).
TIME_SET=$D_TIME_SET
NSTEPS_SRC=$D_NSTEPS_SRC

# ---------------------------------------------------------------------
# Resolve effective config (yaml-derived defaults ← env overrides)
# ---------------------------------------------------------------------
# MIT on/off: env MIT wins, else yaml mit_comparison.enabled.
if [ -n "${MIT:-}" ]; then MIT_ON=$MIT; else MIT_ON=$D_MIT_ENABLED; fi
# Every stage reads the same value: stage 1a writes the MITgcm parameters and
# stage 4 picks the comparison or model-only route from it.
MIT_SET="--set mit_comparison.enabled=$([ "$MIT_ON" -eq 1 ] && echo true || echo false)"

# Placement
LOCOMGCM_NPROC=${LOCOMGCM_NPROC:-1}
LOCOMGCM_NNODES=${LOCOMGCM_NNODES:-1}
case "$LOCOMGCM_NPROC" in ''|*[!0-9]*) echo "[run_case] LOCOMGCM_NPROC must be a positive integer, got '$LOCOMGCM_NPROC'" >&2; exit 2;; esac
case "$LOCOMGCM_NNODES" in ''|*[!0-9]*) echo "[run_case] LOCOMGCM_NNODES must be a positive integer, got '$LOCOMGCM_NNODES'" >&2; exit 2;; esac
if [ "$LOCOMGCM_NPROC" -lt 1 ] || [ "$LOCOMGCM_NNODES" -lt 1 ]; then
    echo "[run_case] LOCOMGCM_NPROC and LOCOMGCM_NNODES must be >= 1 (got LOCOMGCM_NPROC=$LOCOMGCM_NPROC LOCOMGCM_NNODES=$LOCOMGCM_NNODES)" >&2
    exit 2
fi
LOCOMGCM_PARTITION=${LOCOMGCM_PARTITION:-}
LOCOMGCM_NODELIST=${LOCOMGCM_NODELIST:-}
LOCOMGCM_GRES=${LOCOMGCM_GRES:-gpu:$LOCOMGCM_NPROC}
LOCOMGCM_CPUS=${LOCOMGCM_CPUS:-2}
LOCOMGCM_MASTER_PORT=${LOCOMGCM_MASTER_PORT:-29500}
LOCOMGCM_RDZV_ID=${LOCOMGCM_RDZV_ID:-$RUN_ID}
LOCOMGCM_DRY_RUN=${LOCOMGCM_DRY_RUN:-0}
LOCOMGCM_WORLD_SIZE=$((LOCOMGCM_NPROC * LOCOMGCM_NNODES))
LOCOMGCM_PREFIX_NPX=""
LOCOMGCM_PREFIX_NPY=""
if [[ " ${LOCOMGCM_LAUNCH_PREFIX:-} " =~ [[:space:]]LOCOMGCM_NPX=([^[:space:]]+) ]]; then LOCOMGCM_PREFIX_NPX=${BASH_REMATCH[1]}; fi
if [[ " ${LOCOMGCM_LAUNCH_PREFIX:-} " =~ [[:space:]]LOCOMGCM_NPY=([^[:space:]]+) ]]; then LOCOMGCM_PREFIX_NPY=${BASH_REMATCH[1]}; fi
LOCOMGCM_LAYOUT_NPX=${LOCOMGCM_NPX:-${LOCOMGCM_PREFIX_NPX:-$D_NPX}}
LOCOMGCM_LAYOUT_NPY=${LOCOMGCM_NPY:-${LOCOMGCM_PREFIX_NPY:-$D_NPY}}
case "$LOCOMGCM_LAYOUT_NPX" in ''|*[!0-9]*) echo "[run_case] LOCOMGCM_NPX must resolve to a positive integer, got '$LOCOMGCM_LAYOUT_NPX'" >&2; exit 2;; esac
case "$LOCOMGCM_LAYOUT_NPY" in ''|*[!0-9]*) echo "[run_case] LOCOMGCM_NPY must resolve to a positive integer, got '$LOCOMGCM_LAYOUT_NPY'" >&2; exit 2;; esac
if [ "$LOCOMGCM_LAYOUT_NPX" -lt 1 ] || [ "$LOCOMGCM_LAYOUT_NPY" -lt 1 ]; then
    echo "[run_case] LOCOMGCM_NPX/LOCOMGCM_NPY must resolve to >= 1 (got $LOCOMGCM_LAYOUT_NPX x $LOCOMGCM_LAYOUT_NPY)" >&2
    exit 2
fi
LOCOMGCM_LAYOUT_WORLD=$((LOCOMGCM_LAYOUT_NPX * LOCOMGCM_LAYOUT_NPY))
LOCOMGCM_DECOMP_EXPLICIT=0
if [ -n "${LOCOMGCM_NPX:-}" ] || [ -n "${LOCOMGCM_NPY:-}" ] || [ -n "$LOCOMGCM_PREFIX_NPX" ] || [ -n "$LOCOMGCM_PREFIX_NPY" ]; then
    LOCOMGCM_DECOMP_EXPLICIT=1
fi
CPU_PARTITION=${CPU_PARTITION:-}
CPU_NODELIST=${CPU_NODELIST:-}
MIT_PARTITION=${MIT_PARTITION:-$CPU_PARTITION}
MIT_NODELIST=${MIT_NODELIST:-$CPU_NODELIST}
CPU_PLACEMENT_ARGS=()
[ -n "$CPU_PARTITION" ] && CPU_PLACEMENT_ARGS+=(--partition="$CPU_PARTITION")
[ -n "$CPU_NODELIST" ] && CPU_PLACEMENT_ARGS+=(--nodelist="$CPU_NODELIST")
MIT_PLACEMENT_ARGS=()
[ -n "$MIT_PARTITION" ] && MIT_PLACEMENT_ARGS+=(--partition="$MIT_PARTITION")
[ -n "$MIT_NODELIST" ] && MIT_PLACEMENT_ARGS+=(--nodelist="$MIT_NODELIST")
LOCOMGCM_PARTITION_ARG=()
[ -n "$LOCOMGCM_PARTITION" ] && LOCOMGCM_PARTITION_ARG+=(--partition="$LOCOMGCM_PARTITION")


# Ranks + resources
MIT_RANKS=${MIT_RANKS:-$D_RANKS}
PREP_MEM=${PREP_MEM:-$D_MEM_PREP}
LOCOMGCM_MEM=${LOCOMGCM_MEM:-$D_MEM_TO}
MIT_MEM=${MIT_MEM:-$D_MEM_MIT}
# Comparison-metrics depth (stage5). Default 0 = fast: only verdict +
# per-variable rmse/corr (what the dashboard + PASS check read). 1 = full
# offline diagnostics (spatial-bin/spectral/temporal/energy breakdown — slow,
# minutes on large grids; only needed for alignment investigation).
METRICS_FULL=${METRICS_FULL:-0}
PREP_CPUS=${PREP_CPUS:-16}
PREP_MIN=${PREP_MIN:-120}
LOCOMGCM_HOURS=${LOCOMGCM_HOURS:-4}
MIT_HOURS=${MIT_HOURS:-4}
POST_HOURS=${POST_HOURS:-2}
POST_MEM=${POST_MEM:-192G}
POST_CPUS=${POST_CPUS:-16}

# Reuse switches
CLEAN=${CLEAN:-0}
SKIP_PREP=${SKIP_PREP:-0}
REUSE_PREP_FROM=${REUSE_PREP_FROM:-}
SKIP_MIT=${SKIP_MIT:-0}
REUSE_MIT_FROM=${REUSE_MIT_FROM:-}

# Optional output cadence override (default: yaml decides)
SNAPSHOT_SET=""
# Integer-seconds cadence wins over float hours: timestep-aligned dump that
# avoids the float-hours rounding silently dropping/adding frames. Set
# SNAPSHOT_SECONDS to a multiple of dt so dumps land exactly on timesteps
# (e.g. SNAPSHOT_SECONDS=15 = every step at dt=15; 30 = every 2 steps).
if [ -n "${SNAPSHOT_SECONDS:-}" ]; then
    SNAPSHOT_SET="--set output.snapshot_interval_seconds=$SNAPSHOT_SECONDS --set output.nc_output_interval_seconds=$SNAPSHOT_SECONDS"
elif [ -n "${SNAPSHOT_HOURS:-}" ]; then
    SNAPSHOT_SET="--set output.snapshot_interval_hours=$SNAPSHOT_HOURS --set output.nc_output_interval_hours=$SNAPSHOT_HOURS"
fi

# Optional content-addressed prep cache (default OFF; see _pipeline_cache.py).
# Harmless to pass to all stages — only stage1a's regrid pipelines are cacheable.
CACHE_SET=""
[ "${CACHE:-0}" = "1" ]         && CACHE_SET="$CACHE_SET --set cache.reuse=true"
[ "${CACHE_VERIFY:-0}" = "1" ]  && CACHE_SET="$CACHE_SET --set cache.verify=true"
[ "${CACHE_REBUILD:-0}" = "1" ] && CACHE_SET="$CACHE_SET --set cache.rebuild=true"

ENV_SETUP=${LOCOMGCM_ENV_SETUP:-}
# MIT source and optfile are resolved lazily by the native export/build consumers.
export MITGCM_GENMAKE_FLAGS=${MITGCM_GENMAKE_FLAGS:--mpi}
export MITGCM_MPI_LAUNCHER=${MITGCM_MPI_LAUNCHER:-mpirun}
MIT_MPI_ENV_FILE=${MIT_MPI_ENV_FILE:-}

# ---------------------------------------------------------------------
# Resolve which stages run (STAGES= is authoritative; else full pipeline
# per the MIT toggle). Each stage routes to its phase; empty phases skip.
# ---------------------------------------------------------------------
has_stage() { case ",$1," in *",$2,"*) return 0;; *) return 1;; esac; }

if [ -n "${STAGES:-}" ]; then
    # Normalize: strip leading/trailing whitespace, collapse spaces around commas,
    # and remove any blank tokens so "1a, 1b , 2" → "1a,1b,2" and " 2 " → "2".
    REQ=$(printf '%s' "$STAGES" | tr -s '[:space:]' ',' | sed 's/^,*//;s/,*$//;s/,,*/,/g')
elif [ "$MIT_ON" -eq 1 ]; then
    REQ="1a,1b,1c,2,3,4,5"
else
    REQ="1a,1b,2,4"        # LOCOMgcm-only default: no MIT, no compare; 4 is the model-only postproc
fi

# Reject unknown stage tokens early (exit 2 = usage error, not pipeline failure).
for _s in $(printf '%s' "$REQ" | tr ',' ' '); do
    case "$_s" in 1a|1b|1c|2|3|4|5) ;; *)
        echo "[run_case] unknown stage '$_s' in STAGES='$REQ' (valid: 1a 1b 1c 2 3 4 5)" >&2; exit 2;;
    esac
done
unset _s

PREP_STAGES=""
for s in 1a 1b 1c; do has_stage "$REQ" "$s" && PREP_STAGES="${PREP_STAGES:+$PREP_STAGES,}$s"; done
POST_STAGES=""
for s in 4 5; do has_stage "$REQ" "$s" && POST_STAGES="${POST_STAGES:+$POST_STAGES,}$s"; done
DO_TO=0;  has_stage "$REQ" 2 && DO_TO=1
DO_MIT=0; has_stage "$REQ" 3 && DO_MIT=1

if [ -z "$PREP_STAGES" ] && [ "$DO_TO" -eq 0 ] && [ "$DO_MIT" -eq 0 ] && [ -z "$POST_STAGES" ]; then
    echo "[run_case] STAGES='$REQ' resolved to no runnable stage (valid: 1a,1b,1c,2,3,4,5)" >&2; exit 2
fi

# Enforce yaml-coherence early (before any SLURM allocation):
#  - grid divisibility whenever an MPI topology is declared (nPx/nPy > 1)
#  - stage1c target_nProcs == nPx*nPy only when MIT (stage3) actually runs
if { [ "$D_NPX" -gt 1 ] || [ "$D_NPY" -gt 1 ]; } && [ "$D_DIV_OK" -ne 1 ]; then
    echo "[run_case] $D_DIV_MSG" >&2; exit 1
fi
if [ "$DO_MIT" -eq 1 ] && [ "$D_TOPO_OK" -ne 1 ]; then
    echo "[run_case] $D_TOPO_MSG" >&2; exit 1
fi
if [ "$DO_TO" -eq 1 ] && { [ "$LOCOMGCM_WORLD_SIZE" -gt 1 ] || [ "$LOCOMGCM_DECOMP_EXPLICIT" -eq 1 ]; } && [ "$LOCOMGCM_LAYOUT_WORLD" -ne "$LOCOMGCM_WORLD_SIZE" ]; then
    echo "[run_case] LOCOMgcm launcher world size $LOCOMGCM_WORLD_SIZE (LOCOMGCM_NPROC=$LOCOMGCM_NPROC * LOCOMGCM_NNODES=$LOCOMGCM_NNODES) must equal LOCOMgcm decomposition $LOCOMGCM_LAYOUT_WORLD (nPx=$LOCOMGCM_LAYOUT_NPX * nPy=$LOCOMGCM_LAYOUT_NPY). Set LOCOMGCM_NPX/LOCOMGCM_NPY to match the launched ranks." >&2
    exit 2
fi

log "=== run_case: $RUN_ID ==="
log "config=$CONFIG  exp=$D_EXP  grid=${D_NX}x${D_NY}x${D_NZ} (${D_CELLS_M}M cells)  dt=${D_DT}s"
log "run length: ${D_NSTEPS} steps  (source: ${NSTEPS_SRC})"
log "stages: prep=[${PREP_STAGES:-none}]  LOCOMgcm=$([ $DO_TO -eq 1 ] && echo yes || echo no)  MIT=$([ $DO_MIT -eq 1 ] && echo yes || echo no)  post=[${POST_STAGES:-none}]"
[ "$DO_MIT" -eq 1 ] && log "MIT ranks=${MIT_RANKS} (nPx*nPy=${D_NPX}x${D_NPY})"
log "mem: prep=${PREP_MEM} to=${LOCOMGCM_MEM} mit=${MIT_MEM}   placement: LOCOMgcm=${LOCOMGCM_PARTITION}/${LOCOMGCM_NODELIST:-any} CPU=${CPU_PARTITION}/${CPU_NODELIST}"
if [ -n "$CACHE_SET" ]; then
    log "prep cache:${CACHE_SET}  (cache.root=$D_CACHE_ROOT)"
    case "$D_CACHE_ROOT" in
        /tmp/*|/dev/shm/*|/tmp|/dev/shm)
            log "WARNING: cache.root is on a node-local path ($D_CACHE_ROOT); the prep cache is reused only between runs that land on the same node (default $CPU_NODELIST) and is invisible to the login node and other nodes, and is lost when /tmp is cleaned. To persist across nodes, point cache.root in the yaml or via --set cache.root= at the shared project disk." ;;
    esac
fi

# Surface the metrics depth (its effect is otherwise buried in the header comments).
[ -n "$POST_STAGES" ] && log "metrics_full=${METRICS_FULL} (0=verdict+rmse only; 1=full spatial/spectral/energy — fills the dashboard Statistics/CTS/FSS/Spectra tabs)"

INVO=$LOG_DIR/resolved_invocation.yaml
OUT_DIR=$ROOT/outputs/data/$RUN_ID
mkdir -p "$OUT_DIR/_pipeline_meta"
INVO_DURABLE=$OUT_DIR/_pipeline_meta/resolved_invocation.yaml
{
    echo "config: $CONFIG"
    echo "run_id: $RUN_ID"
    echo "git_head: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "recorded_at: $(date -Iseconds)"
    echo "env:"
    for k in STAGES MIT STEPS SECS SNAPSHOT_SECONDS SNAPSHOT_HOURS METRICS_FULL CACHE FUSED \
             IVADV TRITON_KPP STATEKPP DEFCHK N2FUSED MITGCM_ROOT MITGCM_OPTFILE LOCOMGCM_COMPILE_TARGET \
             LOCOMGCM_PARTITION LOCOMGCM_NODELIST LOCOMGCM_NPROC LOCOMGCM_NNODES LOCOMGCM_GRES LOCOMGCM_CPUS LOCOMGCM_MEM LOCOMGCM_HOURS \
             LOCOMGCM_LAUNCH_PREFIX LOCOMGCM_RDZV_ENDPOINT LOCOMGCM_MASTER_PORT LOCOMGCM_RDZV_ID LOCOMGCM_NPX LOCOMGCM_NPY \
             LOCOMGCM_NODE_RANK \
             PREP_MEM PREP_CPUS PREP_MIN \
             MIT_MEM MIT_RANKS MIT_HOURS MIT_PARTITION MIT_NODELIST MIT_MPI_ENV_FILE MITGCM_MPI_LAUNCHER \
             CPU_PARTITION CPU_NODELIST \
             POST_HOURS POST_MEM POST_CPUS; do
        v=${!k:-}
        [ -n "$v" ] && echo "  $k: \"$v\""
    done
} > "$INVO"
cp "$INVO" "$INVO_DURABLE"

# Show the current on-disk run state BEFORE acting — stale / dead-pid / incomplete
# are surfaced here instead of forcing the operator to read manifest.yaml by hand.
if [ -f "$ROOT/outputs/data/$RUN_ID/manifest.yaml" ]; then
    log "current run state (pre-run):"
    "$PYBIN" -m experiments.pipeline_v2.show_run_status "$RUN_ID" 2>/dev/null | sed 's/^/    /' | tee -a "$MASTER_LOG" || true
fi

# ---------------------------------------------------------------------
# Cleanup trap: never orphan the background preparation SLURM steps.
# ---------------------------------------------------------------------
JOB_TO=""
JOB_MIT=""
cleanup() {
    [ -n "$JOB_TO" ]  && kill -0 "$JOB_TO"  2>/dev/null && { echo "[cleanup] killing LOCOMgcm srun $JOB_TO";  kill "$JOB_TO"  2>/dev/null; } || true
    [ -n "$JOB_MIT" ] && kill -0 "$JOB_MIT" 2>/dev/null && { echo "[cleanup] killing MIT srun $JOB_MIT"; kill "$JOB_MIT" 2>/dev/null; } || true
}
trap cleanup EXIT INT TERM

# Build the inner `python -m run_pipeline` command for a given stage list.
# $1 = stages (e.g. "1a,1b,1c"); $2 = optional extra --set (e.g. device=cpu).
pipeline_inner() {
    local pycmd="${4:-$PYBIN}"
    cat <<INNER
set -e
cd $ROOT
$ENV_SETUP
$( [ "$1" = "3" ] && [ -n "$MIT_MPI_ENV_FILE" ] && printf 'source %q\n' "$MIT_MPI_ENV_FILE" )
if [ -x $ROOT/bin/jit_lock_sweep.sh ]; then $ROOT/bin/jit_lock_sweep.sh; fi
export LOCOMGCM_METRICS_FULL=$METRICS_FULL
${3:-} $pycmd -m experiments.pipeline_v2.run_pipeline \
  --config "$CONFIG" --run-id "$RUN_ID" --output $ROOT/outputs/data/ \
  --set "run.experiment_name=$RUN_ID" \
  $MIT_SET $TIME_SET $SNAPSHOT_SET $CACHE_SET ${2:-} \
  --stages $1
echo '== stages $1 done =='
INNER
}

# ---------------------------------------------------------------------
# Optional clean
# ---------------------------------------------------------------------
if [ "$CLEAN" -eq 1 ]; then
    OUT_DIR=$ROOT/outputs/data/$RUN_ID
    if [ -d "$OUT_DIR" ] || [ -L "$OUT_DIR" ]; then
        log "CLEAN=1 — removing $OUT_DIR"
        rm -rf "$OUT_DIR"
    fi
fi

guard_mitgcm_symlink_before_1c() {
    has_stage "$PREP_STAGES" 1c || return 0
    local mit_dir=$ROOT/outputs/data/$RUN_ID/mitgcm
    [ -L "$mit_dir" ] || return 0
    if [ "$SKIP_MIT" -eq 1 ]; then
        log "outputs/data/$RUN_ID/mitgcm is a symlink to $(readlink "$mit_dir"); removing it before stage1c (SKIP_MIT re-links it after the LOCOMgcm launch)"
        rm -f "$mit_dir"
    else
        echo "[run_case] outputs/data/$RUN_ID/mitgcm is a symlink to $(readlink "$mit_dir"); stage1c would rewrite that run's MIT inputs. Remove the link first, or set SKIP_MIT=1 REUSE_MIT_FROM=<run-id>." >&2
        exit 1
    fi
}

if [ -n "$PREP_STAGES" ]; then
guard_mitgcm_symlink_before_1c
if [ "$SKIP_PREP" -eq 1 ]; then
    if [ -z "$REUSE_PREP_FROM" ]; then
        log "SKIP_PREP=1 requires REUSE_PREP_FROM=<run-id>"; exit 2
    fi
    SRC_RUN=$ROOT/outputs/data/$REUSE_PREP_FROM
    DST_RUN=$ROOT/outputs/data/$RUN_ID
    mkdir -p "$DST_RUN/_pipeline_meta"
    # Reuse dt-independent stage1a/1b artifacts via symlink; stage1c (dt-dependent
    # MIT namelists) reruns below. model_input large files are symlinked but
    # run_params.yaml is copied + synced (stage1c reads output.* from it, and a
    # dt change must propagate).
    for sub in torch_inputs preprocessed_cf; do
        [ -e "$DST_RUN/_pipeline_meta/$sub" ] || ln -s "$SRC_RUN/_pipeline_meta/$sub" "$DST_RUN/_pipeline_meta/$sub"
    done
    # config_snapshot/source_config.yaml is read by the comparison adapter's
    # load_grid -> _geo_axes_from_config to recover geographic lat/lon (the LOCOMgcm
    # NC writer stores index/metre axes). Without it the adapter keeps
    # non-geographic coords and cross-run grid_signature/diff breaks. The geo
    # grid is identical across runs of the same case, so reuse the source's.
    [ -e "$DST_RUN/config_snapshot" ] || ln -s "$SRC_RUN/config_snapshot" "$DST_RUN/config_snapshot"
    mkdir -p "$DST_RUN/_pipeline_meta/stage2_torch_run"
    "$PYBIN" <<PYEOF | tee -a "$MASTER_LOG"
import yaml
from pathlib import Path
src_model = Path("$SRC_RUN/_pipeline_meta/model_input")
dst_model = Path("$DST_RUN/_pipeline_meta/model_input")
cfg = yaml.safe_load(open("$CONFIG"))
if dst_model.is_symlink(): dst_model.unlink()
dst_model.mkdir(parents=True, exist_ok=True)
for child in src_model.iterdir():
    if child.name == "run_params.yaml": continue
    link = dst_model / child.name
    if not (link.exists() or link.is_symlink()):
        link.symlink_to(child.resolve())
yt = cfg.get("time", {})
yaml_dt = float(yt.get("dt"))
params = yaml.safe_load(open(src_model / "run_params.yaml"))
baked_dt = float(params.get("fixed_dt", params.get("dt_clock", yaml_dt)))
if abs(baked_dt - yaml_dt) > 1e-9:
    print(f"[run_params] dt mismatch baked={baked_dt} yaml={yaml_dt} — rewriting")
    for key, yk in [("fixed_dt","dt"),("dt_mom","dt_mom"),("dt_tracer","dt_tracer"),("dt_freesurf","dt_freesurf"),("dt_clock","dt_clock")]:
        if yk in yt: params[key] = float(yt[yk])
yout = cfg.get("output", {})
for ok in ("nc_output_interval_hours","snapshot_interval_hours","monitor_interval_steps"):
    if ok in yout: params[ok] = yout[ok]
yaml.safe_dump(params, open(dst_model / "run_params.yaml", "w"), default_flow_style=False, sort_keys=False)
# Sanitize manifest so stage prereq checks pass with reused stages.
m = yaml.safe_load(open("$SRC_RUN/manifest.yaml"))
m["run_id"] = "$RUN_ID"
m["stages"].pop("stage2_torch_run", None)
m["stages"].pop("stage1c_export_mit", None)
yaml.safe_dump(m, open("$DST_RUN/manifest.yaml", "w"), default_flow_style=False, sort_keys=False)
print(f"[manifest] reused from $REUSE_PREP_FROM; stages kept: {list(m['stages'].keys())}")
PYEOF
    if has_stage "$PREP_STAGES" 1c; then
        log "SKIP_PREP=1 — reused 1a/1b from $REUSE_PREP_FROM; rerunning stage1c"
        srun "${CPU_PLACEMENT_ARGS[@]}" \
             --cpus-per-task=4 --mem=32G --time=15:00 --job-name="${RUN_ID}-1c" \
             bash -lc "$(pipeline_inner "1c" "--set run.device=cpu")" \
             2>&1 | tee "$PREP_LOG" | tee -a "$MASTER_LOG"
        # pipefail already fires on srun failure; this is belt-and-suspenders if pipefail is ever disabled
        [ "${PIPESTATUS[0]}" -ne 0 ] && { log "stage1c FAILED — see $PREP_LOG"; exit 1; }
    else
        log "SKIP_PREP=1 — reused prep from $REUSE_PREP_FROM; no prep stage to rerun"
    fi
else
    log "prep ($PREP_STAGES) srun (partition=$CPU_PARTITION node=$CPU_NODELIST cpus=$PREP_CPUS mem=$PREP_MEM time=${PREP_MIN}m)"
    srun "${CPU_PLACEMENT_ARGS[@]}" \
         --cpus-per-task="$PREP_CPUS" --mem="$PREP_MEM" --time="${PREP_MIN}:00" --job-name="${RUN_ID}-prep" \
         bash -lc "$(pipeline_inner "$PREP_STAGES" "--set run.device=cpu")" \
         2>&1 | tee "$PREP_LOG" | tee -a "$MASTER_LOG"
    # pipefail already fires on srun failure; this is belt-and-suspenders if pipefail is ever disabled
    [ "${PIPESTATUS[0]}" -ne 0 ] && { log "prep FAILED — see $PREP_LOG"; exit 1; }
fi
else
    log "skipped (no prep stage requested)"
fi

# ---------------------------------------------------------------------
# LOCOMgcm and/or MIT (whichever of stage 2 / 3 was requested)
# ---------------------------------------------------------------------
JOB_TO=""; JOB_MIT=""
RUN_TO=0; RUN_MIT=0

if [ "$DO_TO" -eq 1 ]; then
    FUSED_PREFIX=""
    [ "${FUSED:-1}" = "1" ] && FUSED_PREFIX="LOCOMGCM_CG3D_FUSED_PCG=1 "

    ACCEL_PREFIX=""
    if [ "$D_DEVICE" = "cuda" ]; then
        [ "${IVADV:-1}" = "1" ] && ACCEL_PREFIX="${ACCEL_PREFIX}LOCOMGCM_IVADV_FUSED_BACKEND=cuda "
        if [ "${DEFCHK:-1}" = "1" ]; then
            ACCEL_PREFIX="${ACCEL_PREFIX}LOCOMGCM_DEFERRED_CHECKS=1 "
        else
            ACCEL_PREFIX="${ACCEL_PREFIX}LOCOMGCM_DEFERRED_CHECKS=0 "
        fi
    elif [ "${IVADV:-1}" = "1" ]; then
        log "NOTE: run.device=$D_DEVICE — fused ivadv NOT injected (CUDA-only kernel; it raises rather than falling back)"
    fi
    [ -n "${TRITON_KPP:-}" ] && log "DEPRECATED: TRITON_KPP lever renamed STATEKPP (Triton impl retired; maps to LOCOMGCM_KPP_STATEKPP_CUDA)"
    [ "${STATEKPP:-${TRITON_KPP:-1}}" = "1" ] && ACCEL_PREFIX="${ACCEL_PREFIX}LOCOMGCM_KPP_STATEKPP_CUDA=1 "
    if [ "${N2FUSED:-0}" = "1" ]; then
        [ "$D_DEVICE" = "cuda" ] || log "NOTE: N2FUSED=1 but run.device=$D_DEVICE — fused kernels fall back to eager off-CUDA"
        ACCEL_PREFIX="${ACCEL_PREFIX}LOCOMGCM_KPP_FUSED_CUDA=1 LOCOMGCM_VIM_FUSED_CUDA=1 LOCOMGCM_TRACER_ADV_FUSED_CUDA=1 "
    fi
    # Checkpoint resume: `resume <id>` sets RESUME_CKPT=latest so the LOCOMgcm stage
    # continues from the newest checkpoint; without it stage2 starts at step 0
    # (checkpoint.resume defaults to none). RESUME_CKPT also works on a normal
    # invocation to continue a previously-killed run.
    RESUME_SET=""
    [ -n "${RESUME_CKPT:-}" ] && RESUME_SET="--set checkpoint.resume=${RESUME_CKPT}"
    LOCOMGCM_PYCMD="$PYBIN"
    LOCOMGCM_LAUNCH_DESC="single"
    LOCOMGCM_TOPOLOGY_ARGS=()
    if [ "$LOCOMGCM_WORLD_SIZE" -gt 1 ]; then
        LOCOMGCM_PYCMD="$PYBIN -m torch.distributed.run --standalone --nproc_per_node=$LOCOMGCM_NPROC --no-python $PYBIN"
        LOCOMGCM_LAUNCH_DESC="torchrun-standalone"
        if [ "$LOCOMGCM_NNODES" -gt 1 ]; then
            LOCOMGCM_PYCMD="$PYBIN -m torch.distributed.run --nnodes=$LOCOMGCM_NNODES --nproc_per_node=$LOCOMGCM_NPROC --node_rank=\${LOCOMGCM_NODE_RANK:-\${SLURM_NODEID:-0}} --rdzv_backend=c10d --rdzv_endpoint=\${LOCOMGCM_RDZV_ENDPOINT:-\$(scontrol show hostnames \"\$SLURM_JOB_NODELIST\" | head -n1):$LOCOMGCM_MASTER_PORT} --rdzv_id=$LOCOMGCM_RDZV_ID --no-python $PYBIN"
            LOCOMGCM_LAUNCH_DESC="torchrun-c10d"
            LOCOMGCM_TOPOLOGY_ARGS+=(--nodes="$LOCOMGCM_NNODES" --ntasks="$LOCOMGCM_NNODES" --ntasks-per-node=1)
        fi
    fi
    log "LOCOMgcm srun (partition=$LOCOMGCM_PARTITION node=${LOCOMGCM_NODELIST:-any} nnodes=$LOCOMGCM_NNODES nproc_per_node=$LOCOMGCM_NPROC world=$LOCOMGCM_WORLD_SIZE launcher=$LOCOMGCM_LAUNCH_DESC cpus=$LOCOMGCM_CPUS mem=$LOCOMGCM_MEM gres=${LOCOMGCM_GRES:-none} time=${LOCOMGCM_HOURS}h)  fused=${FUSED:-1} accel=[${ACCEL_PREFIX:-none}]${RESUME_CKPT:+  resume=$RESUME_CKPT}"
    LOCOMGCM_GRES_ARG=(); [ -n "$LOCOMGCM_GRES" ] && LOCOMGCM_GRES_ARG=(--gres="$LOCOMGCM_GRES")
    LOCOMGCM_NODELIST_ARG=(); [ -n "$LOCOMGCM_NODELIST" ] && LOCOMGCM_NODELIST_ARG=(--nodelist="$LOCOMGCM_NODELIST")
    LOCOMGCM_SRUN_CMD=(srun "${LOCOMGCM_PARTITION_ARG[@]}" "${LOCOMGCM_NODELIST_ARG[@]}" "${LOCOMGCM_TOPOLOGY_ARGS[@]}" \
         --cpus-per-task="$LOCOMGCM_CPUS" --mem="$LOCOMGCM_MEM" --time="${LOCOMGCM_HOURS}:00:00" "${LOCOMGCM_GRES_ARG[@]}" \
         --job-name="${RUN_ID}-to")
    LOCOMGCM_INNER=$(pipeline_inner "2" "$RESUME_SET" "${FUSED_PREFIX}${ACCEL_PREFIX}${LOCOMGCM_LAUNCH_PREFIX:-}" "$LOCOMGCM_PYCMD")
    if [ "$LOCOMGCM_DRY_RUN" = "1" ]; then
        log "LOCOMGCM_DRY_RUN=1 — not launching LOCOMgcm srun"
        printf 'LOCOMGCM_DRY_RUN_SRUN=' | tee -a "$MASTER_LOG"
        printf '%q ' "${LOCOMGCM_SRUN_CMD[@]}" | tee -a "$MASTER_LOG"
        printf '\n' | tee -a "$MASTER_LOG"
        printf 'LOCOMGCM_DRY_RUN_INNER<<EOF\n%s\nEOF\n' "$LOCOMGCM_INNER" | tee -a "$MASTER_LOG"
    else
        "${LOCOMGCM_SRUN_CMD[@]}" bash -lc "$LOCOMGCM_INNER" > "$LOCOMGCM_LOG" 2>&1 &
        JOB_TO=$!; RUN_TO=1
        log "LOCOMgcm pid=$JOB_TO log=$LOCOMGCM_LOG"
    fi
fi

if [ "$DO_MIT" -eq 1 ]; then
    if [ "$SKIP_MIT" -eq 1 ]; then
        if [ -z "$REUSE_MIT_FROM" ]; then
            log "SKIP_MIT=1 requires REUSE_MIT_FROM=<run-id>"; exit 2
        fi
        SRC=$ROOT/outputs/data/$REUSE_MIT_FROM/mitgcm
        DST=$ROOT/outputs/data/$RUN_ID/mitgcm
        if [ -d "$SRC" ]; then
            rm -rf "$DST" 2>/dev/null || true
            ln -s "$SRC" "$DST"
            log "Symlinked MIT from $REUSE_MIT_FROM (stage3 skipped)"
        else
            echo "[run_case] SKIP_MIT: $SRC not found — cannot reuse; aborting (set STAGES to skip 4,5 if comparison is not needed)" >&2; exit 1
        fi
    else
        log "MIT srun (partition=$MIT_PARTITION node=$MIT_NODELIST cpus=$MIT_RANKS mem=$MIT_MEM time=${MIT_HOURS}h, NUMA-local)"
        srun "${MIT_PLACEMENT_ARGS[@]}" \
             --cpus-per-task="$MIT_RANKS" --mem="$MIT_MEM" --time="${MIT_HOURS}:00:00" \
             --hint=nomultithread --cpu-bind=cores --distribution=block:block \
             --job-name="${RUN_ID}-mit" \
             bash -lc "$(pipeline_inner "3")" > "$MIT_LOG" 2>&1 &
        JOB_MIT=$!; RUN_MIT=1
        log "MIT pid=$JOB_MIT log=$MIT_LOG"
    fi
fi

if [ "$RUN_TO" -eq 1 ] || [ "$RUN_MIT" -eq 1 ]; then
    # Monitor loop: reap whichever was launched. If LOCOMgcm fails while MIT runs,
    # cancel MIT early (comparison impossible); if MIT fails, let LOCOMgcm finish.
    # set +e so a nonzero `wait` doesn't trip set -e before we capture rc.
    set +e
    RC_TO=""; RC_MIT=""
    [ "$RUN_TO" -eq 0 ] && RC_TO=0
    [ "$RUN_MIT" -eq 0 ] && RC_MIT=0
    while [ -z "$RC_TO" ] || [ -z "$RC_MIT" ]; do
        if [ "$RUN_TO" -eq 1 ] && [ -z "$RC_TO" ] && ! kill -0 "$JOB_TO" 2>/dev/null; then
            wait "$JOB_TO"; RC_TO=$?
            if [ "$RC_TO" -ne 0 ] && [ "$RUN_MIT" -eq 1 ] && [ -z "$RC_MIT" ] && kill -0 "$JOB_MIT" 2>/dev/null; then
                log "LOCOMgcm failed (rc=$RC_TO) → cancelling MIT"
                kill "$JOB_MIT" 2>/dev/null
            fi
        fi
        if [ "$RUN_MIT" -eq 1 ] && [ -z "$RC_MIT" ] && ! kill -0 "$JOB_MIT" 2>/dev/null; then
            wait "$JOB_MIT"; RC_MIT=$?
        fi
        [ -z "$RC_TO" ] || [ -z "$RC_MIT" ] && sleep 5
    done
    set -e
    JOB_TO=""; JOB_MIT=""   # reaped; disarm trap kills
    log "finished: LOCOMgcm rc=$RC_TO  MIT rc=$RC_MIT"
    if { [ "$RUN_TO" -eq 1 ] && [ "$RC_TO" -ne 0 ]; } || { [ "$RUN_MIT" -eq 1 ] && [ "$RC_MIT" -ne 0 ]; }; then
        log "FAILED"
        exit 1
    fi
else
    log "skipped (no LOCOMgcm/MIT stage requested)"
fi

# ---------------------------------------------------------------------
# postproc stages (subset of 4,5) on CPU node
# ---------------------------------------------------------------------
if [ -n "$POST_STAGES" ]; then
    log "postproc ($POST_STAGES) srun (partition=$CPU_PARTITION node=$CPU_NODELIST cpus=${POST_CPUS} mem=${POST_MEM} time=${POST_HOURS}h)"
    srun "${CPU_PLACEMENT_ARGS[@]}" \
         --cpus-per-task="$POST_CPUS" --mem="$POST_MEM" --time="${POST_HOURS}:00:00" --job-name="${RUN_ID}-post" \
         bash -lc "$(pipeline_inner "$POST_STAGES" "--set run.device=cpu")" \
         2>&1 | tee "$POST_LOG" | tee -a "$MASTER_LOG"
    # pipefail already fires on srun failure; this is belt-and-suspenders if pipefail is ever disabled
    [ "${PIPESTATUS[0]}" -ne 0 ] && { log "postproc FAILED — see $POST_LOG"; exit 1; }
fi

log "=== run_case END (all requested stages complete) ==="
exit 0
