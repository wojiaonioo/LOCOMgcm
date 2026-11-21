#!/bin/bash
# =============================================================
# MITgcm comparison run: build and execute from exported case
#
# Usage, in an environment that provides the MITgcm compiler and MPI:
#   bash experiments/run_mitgcm_comparison.sh <case_dir> <mitgcm_root>
#
# Arguments:
#   case_dir    — MITgcm case directory containing code/, input/
#   mitgcm_root — MITgcm source root directory (with tools/, model/, pkg/)
# =============================================================
set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <case_dir> <mitgcm_root> [n_procs]"
    echo "  case_dir:    MITgcm case directory (code/, input/, build/, run/)"
    echo "  mitgcm_root: MITgcm source root (tools/, model/, pkg/)"
    echo "  n_procs:     MPI rank count (default 1 → ./mitgcmuv; >1 → mpirun -np N)"
    exit 1
fi

CASE_DIR="$(cd "$1" && pwd)"
# This stdlib-only CLI also works when the native environment has no torch.
EXTERNAL_PATHS_CLI="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/locomgcm/external_paths.py"
MITGCM_ROOT=$("${LOCOMGCM_PYTHON:-python}" "$EXTERNAL_PATHS_CLI" mitgcm_root --value "$2")
NPROCS="${3:-1}"
BUILD_DIR="$CASE_DIR/build"
RUN_DIR="$CASE_DIR/run"
# OPTFILE / genmake flags / launcher are env-driven so different hosts can
# pick their own MPI toolchain without forking this script.
#   MITGCM_OPTFILE          — full path or configured mitgcm_optfile
#   MITGCM_GENMAKE_FLAGS    — extra genmake2 flags (e.g. "-mpi")
#   MITGCM_MPI_LAUNCHER     — mpirun-compatible launcher (default "mpirun")
#   MITGCM_MPI_OVERSUBSCRIBE — 1 (default) passes --oversubscribe; 0 drops it.
#                             Only needed when SLURM_NTASKS < NPROCS.
#   MITGCM_MPI_BIND         — --bind-to value (default "none"; use "core" for
#                             performance runs on a correct --ntasks alloc)
#   MITGCM_MPI_MAP_BY       — optional --map-by value (unset = OpenMPI default)
OPTFILE=$("${LOCOMGCM_PYTHON:-python}" "$EXTERNAL_PATHS_CLI" mitgcm_optfile)
GENMAKE_FLAGS="${MITGCM_GENMAKE_FLAGS:-}"
MPI_LAUNCHER="${MITGCM_MPI_LAUNCHER:-mpirun}"

echo "========================================"
echo " MITgcm Comparison Run"
echo "========================================"
echo "CASE_DIR:       $CASE_DIR"
echo "MITGCM_ROOT:    $MITGCM_ROOT"
echo "OPTFILE:        $OPTFILE"
echo "GENMAKE_FLAGS:  ${GENMAKE_FLAGS:-(none)}"
echo "MPI_LAUNCHER:   $MPI_LAUNCHER"
echo "NPROCS:         $NPROCS"
echo ""

# Verify prerequisites
if [ ! -d "$CASE_DIR/code" ]; then
    echo "ERROR: $CASE_DIR/code not found"
    exit 1
fi
if [ ! -d "$CASE_DIR/input" ]; then
    echo "ERROR: $CASE_DIR/input not found"
    exit 1
fi
if [ ! -f "$OPTFILE" ]; then
    echo "ERROR: optfile not found: $OPTFILE"
    exit 1
fi

# Check compiler
echo "Compiler: $(ifx --version 2>&1 | head -1 || echo 'ifx not found')"
echo ""

# ---- Step 1: Clean build directory ----
echo "[Step 1] Cleaning build directory ..."
cd "$BUILD_DIR"
rm -rf *.o *.f *.F *.c *.h *.p Makefile mitgcmuv genmake* Make.depend \
       build_options.* AD_CONFIG.* packages.* 2>/dev/null || true
echo "  -> Build directory cleaned"

# ---- Step 2: genmake2 ----
echo "[Step 2] Running genmake2 (flags='${GENMAKE_FLAGS:-none}') ..."
"$MITGCM_ROOT/tools/genmake2" \
    -mods "$CASE_DIR/code" \
    -of "$OPTFILE" \
    -rootdir "$MITGCM_ROOT" \
    ${GENMAKE_FLAGS} > genmake2.log 2>&1 || { cat genmake2.log; exit 1; }
echo "  -> genmake2 done"

# ---- Step 3: make depend ----
echo "[Step 3] make depend ..."
make depend > make_depend.log 2>&1 || { cat make_depend.log; exit 1; }
echo "  -> make depend done"

# ---- Step 4: make ----
NPROC=$(nproc 2>/dev/null || echo 4)
echo "[Step 4] make -j${NPROC} ..."
make -j"${NPROC}" > make.log 2>&1 || { cat make.log; exit 1; }
echo "  -> Build complete: $(ls -la mitgcmuv 2>/dev/null || echo 'FAILED')"

if [ ! -f "mitgcmuv" ]; then
    echo "ERROR: mitgcmuv binary not built"
    exit 1
fi

# ---- Step 5: Prepare run directory ----
echo "[Step 5] Preparing run directory ..."
cd "$RUN_DIR"
rm -f *.bin *.data *.meta data* eedata STDOUT* mitgcmuv 2>/dev/null || true
ln -sf ../input/* .
ln -sf ../build/mitgcmuv .
echo "  -> Run directory prepared"

# ---- Step 6: Run ----
echo "[Step 6] Running MITgcm (n_procs=$NPROCS via $MPI_LAUNCHER) ..."
# MITgcm is always built with -mpi (MITGCM_GENMAKE_FLAGS) so the binary
# calls MPI_INIT even when NPROCS=1.  Launching ./mitgcmuv directly under
# srun --ntasks=1 then hits OpenMPI's PMI bootstrap (OPAL ERROR
# pmix3x_client.c:112 "Unreachable") because the system OpenMPI was not
# built with SLURM PMIx support.  Routing every launch through mpirun
# (even -np 1) sidesteps the SLURM PMI dependency entirely.
#
# --oversubscribe is required under srun --cpus-per-task=N (SLURM_NTASKS=1):
#   OpenMPI's slurm RAS plugin then reports just 1 slot and refuses
#   -np N.  --oversubscribe is the official bypass and the error
#   message itself recommends it.  --host alone does NOT help because
#   slurm RAS overrides command-line --host.  (Verified with a 4-test
#   matrix on a SLURM cluster.)
# --bind-to none avoids OpenMPI auto-binding clashing with slurm cgroup
#   cpuset.
#
# Both flags are compensations for SLURM_NTASKS=1 and cost real performance
# once the allocation is correct: --oversubscribe makes OpenMPI enter the
# yield-when-idle degraded mode (waiting ranks call sched_yield, turning
# sub-100us allreduces into millisecond-scale ones), and --bind-to none lets
# every rank drift across the whole node, destroying NUMA and cache locality.
# On a job that really does request --ntasks=N neither is needed.  They stay
# on by default so existing callers are untouched; benchmark runs override:
#   MITGCM_MPI_OVERSUBSCRIBE=0 MITGCM_MPI_BIND=core MITGCM_MPI_MAP_BY=core
#
# Split mpirun stderr to mpirun.err so launch failures aren't buried
#   inside MITgcm's STDOUT.0001 (faster future debugging).
MPI_EXTRA=()
[ "${MITGCM_MPI_OVERSUBSCRIBE:-1}" = "1" ] && MPI_EXTRA+=(--oversubscribe)
MPI_EXTRA+=(--bind-to "${MITGCM_MPI_BIND:-none}")
[ -n "${MITGCM_MPI_MAP_BY:-}" ] && MPI_EXTRA+=(--map-by "$MITGCM_MPI_MAP_BY")
echo "  mpirun flags: ${MPI_EXTRA[*]}"
"$MPI_LAUNCHER" -np "$NPROCS" "${MPI_EXTRA[@]}" \
    ./mitgcmuv > STDOUT.0001 2> mpirun.err
echo "  -> Run complete"

# ---- Step 7: Result check ----
OUTPUT="$RUN_DIR/STDOUT.0001"
echo ""
echo "========================================"
echo " Run Results"
echo "========================================"
if grep -q "PROGRAM MAIN: Execution ended Normally" "$OUTPUT"; then
    echo "  ✅  MITgcm finished normally"
else
    echo "  ❌  MITgcm may have failed — check STDOUT.0001"
    echo "  Last 20 lines:"
    tail -20 "$OUTPUT"
fi

echo ""
echo "  Monitor (last step):"
grep "dynstat_eta_max"   "$OUTPUT" | tail -1 || true
grep "dynstat_eta_min"   "$OUTPUT" | tail -1 || true
grep "dynstat_uvel_max"  "$OUTPUT" | tail -1 || true
grep "dynstat_theta_max" "$OUTPUT" | tail -1 || true
grep "dynstat_theta_min" "$OUTPUT" | tail -1 || true

# Count output files
N_ETA=$(ls -1 "$RUN_DIR"/Eta.*.data 2>/dev/null | wc -l)
N_T=$(ls -1 "$RUN_DIR"/T.*.data 2>/dev/null | wc -l)
echo ""
echo "  Output files: ${N_ETA} Eta snapshots, ${N_T} T snapshots"
echo "  STDOUT: $OUTPUT"
echo ""
echo "========================================"
echo " Done!"
echo "========================================"
