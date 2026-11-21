#!/usr/bin/env bash
# Run the regional forward or adjoint executable with the prepared inputs.
set -euo pipefail
: "${SLURM_JOB_ID:?Model runs require a compute allocation}"

here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"

name=${1:?usage: run_mit.sh RUN_NAME fwd|adj BUILD_DIR}
mode=${2:?missing MODE (fwd|adj)}
build=$(cd "${3:?missing BUILD_DIR}" && pwd)
case "$mode" in
  fwd) exe=mitgcmuv ;;
  adj) exe=mitgcmuv_tap_adj ;;
  *) echo "mode must be fwd or adj"; exit 2 ;;
esac
[ -x "$build/build/$exe" ] || { echo "missing $build/build/$exe"; exit 2; }

rd=$RPAIR_WORK/runs/$name
[ ! -e "$rd" ] || { echo "Refuse to overwrite $rd"; exit 2; }
mkdir -p "$rd"
rd=$(cd "$rd" && pwd)

# ---- reference inputs: binaries linked, text files copied ---------------------
for f in "$RPAIR_CASE/input"/*; do
  b=$(basename "$f")
  case "$b" in
    *.bin|*.meta) ln -s "$(readlink -f "$f")" "$rd/$b" ;;
    *) cp "$f" "$rd/$b" ;;
  esac
done
ln -s "$build/build/$exe" "$rd/$exe"

# ---- data.pkg: diagnostics off (package not compiled), ctrl/autodiff on ------
sed -i 's/^ useDiagnostics=\.TRUE\.,$/ useDiagnostics=.FALSE.,\n useCTRL=.TRUE.,\n useAUTODIFF=.TRUE.,\n useGrdchk=.FALSE.,/' "$rd/data.pkg"
grep -q '^ useDiagnostics=\.FALSE\.,$' "$rd/data.pkg"
grep -q '^ useCTRL=\.TRUE\.,$' "$rd/data.pkg"

# ---- cost and control ----------------------------------------------------------
cat > "$rd/data.cost" <<'NML'
 &COST_NML
 mult_test=1.,
 &
NML
# Two generic 3-D controls, unit weights (CTRL_MAP_GENARR3D divides by SQRT(w),
# pkg/ctrl/ctrl_map_genarr.F:349-351), no penalty (mult_genarr3d = 0, so
# useCtrlCostContribution stays .FALSE., ctrl_readparms.F:506-518), no bounds
# (xx_genarr3d_bounds default 0, the gate at ctrl_bound.F:41 is 0 < 0).
cat > "$rd/data.ctrl" <<'NML'
 &CTRL_NML
 doInitXX=.TRUE.,
 doMainUnpack=.FALSE.,
 doMainPack=.FALSE.,
 ctrlprec=64,
 &
 &CTRL_PACKNAMES
 &
 &CTRL_NML_GENARR
 xx_genarr3d_file(1)='xx_theta',
 xx_genarr3d_weight(1)='ones_64b.bin',
 mult_genarr3d(1)=0.,
 xx_genarr3d_file(2)='xx_salt',
 xx_genarr3d_weight(2)='ones_64b.bin',
 mult_genarr3d(2)=0.,
 &
NML
cat > "$rd/data.optim" <<'NML'
 &OPTIM
 optimcycle=0,
 &
NML
# Every value is the source default (pkg/autodiff/autodiff_readparms.F), written out so the
# adjoint switches are on record: exact KPP and advection in the adjoint, self-adjoint cg2d.
cat > "$rd/data.autodiff" <<'NML'
 &AUTODIFF_PARM01
 inAdExact=.TRUE.,
 cg2dFullAdjoint=.FALSE.,
 useKPPinAdMode=.TRUE.,
 useApproxAdvectionInAdMode=.FALSE.,
 viscFacInAd=1.,
 viscFacInFw=1.,
 &
NML
"$RPAIR_PY" - "$rd" <<'PY'
import sys
from pathlib import Path
import numpy as np
np.ones((90, 152, 160), dtype=">f8").tofile(Path(sys.argv[1]) / "ones_64b.bin")
PY

# ---- provenance (no digests) ------------------------------------------------------
{
  echo "run_name  $name"
  echo "slurm_job $SLURM_JOB_ID"
  echo "host      $(hostname)"
  echo "mode      $mode ($exe)"
  echo "build     $build"
  echo "case_ref  $RPAIR_CASE"
  ls -l "$build/build/$exe"
  echo "--- namelists against the reference input/ ---"
  for f in data data.cal data.exf data.kpp data.obcs eedata data.diagnostics; do
    cmp "$RPAIR_CASE/input/$f" "$rd/$f" && echo "same $f"
  done
  diff "$RPAIR_CASE/input/data.pkg" "$rd/data.pkg" || true
} > "$rd/provenance.txt" 2>&1

# ---- run, using MPI (40 ranks, cores bound, UCX) ----------------------
cd "$rd"
source "$RPAIR_REPO/bin/mit_fat_mpi_env.sh"
set +e
"$RPAIR_MPIRUN" -np "$RPAIR_NPROCS" --oversubscribe --bind-to core --map-by socket \
    "./$exe" > run.out 2> mpirun.err
rc=$?
set -e
echo "mpirun rc=$rc"
if grep -q 'PROGRAM MAIN: Execution ended Normally' STDOUT.0000 2>/dev/null; then
  echo "RUN_OK $rd"
else
  echo "RUN_FAILED $rd"
  tail -40 STDOUT.0000 2>/dev/null || true
  tail -20 STDERR.0000 2>/dev/null || true
  tail -20 run.out || true
  tail -20 mpirun.err || true
  exit 7
fi
grep -E '^fc *=' costfunction.0000 2>/dev/null || echo "(no costfunction.0000)"
ls -l adxx_* 2>/dev/null || true
