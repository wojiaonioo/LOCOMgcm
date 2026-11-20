#!/usr/bin/env bash
# Build the regional forward and Tapenade adjoint executables.
set -euo pipefail
: "${SLURM_JOB_ID:?Build requires a compute allocation}"

here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"

out=${1:?usage: build.sh OUTPUT_DIR}
mkdir -p "$out"
out=$(cd "$out" && pwd)
[ ! -e "$out/build" ] || { echo "Refuse to overwrite $out/build"; exit 2; }
mkdir -p "$out/code" "$out/build"

# ---- 1. forward-reference headers, unchanged --------------------------------
# The expl reference has no mom_vecinv.F in code/: it uses the upstream file.
[ ! -e "$RPAIR_CASE/code/mom_vecinv.F" ] || { echo "reference code/ has mom_vecinv.F"; exit 2; }
cp -L "$RPAIR_CASE/code/SIZE.h" "$RPAIR_CASE/code/CPP_OPTIONS.h" \
      "$RPAIR_CASE/code/EXF_OPTIONS.h" "$RPAIR_CASE/code/OBCS_OPTIONS.h" "$out/code/"

# ---- 2. edits derived from the build tree, with exact-match checks ----------
"$RPAIR_PY" "$here/make_tree_edits.py" "$RPAIR_MIT_ROOT" "$out/code"

# ---- 3. adjoint source overrides ----------------------------------------------------
cp "$here/overrides/COST_OPTIONS.h" "$here/overrides/cost_test.F" \
   "$here/overrides/cost_tile.F" "$here/overrides/RPAIR_COST_SECTION.h" "$out/code/"
cp "$here/overrides/packages.conf.txt" "$out/code/packages.conf"

# Tapenade's stack spills to tapStack%05i in the current directory beyond
# ADSTACK_MAX_SPACES 1 MiB blocks (pkg/tapenade/adStack.c:56, :998-1010); 40 ranks share the
# run directory, so the cap is raised through the C compiler.
sed -e 's|^CFLAGS=\(.*\)$|CFLAGS=\1\nCFLAGS="$CFLAGS -DADSTACK_MAX_SPACES=200000"|' \
    "$RPAIR_OPTFILE" > "$out/optfile"
grep -q 'DADSTACK_MAX_SPACES=200000' "$out/optfile"

# ---- 4. identity (file listing and diffs, no digests) ------------------------
{
  echo "host      $(hostname)"
  echo "slurm_job $SLURM_JOB_ID"
  echo "mit_tree  $RPAIR_MIT_ROOT"
  echo "case_ref  $RPAIR_CASE"
  echo "optfile   $RPAIR_OPTFILE (+ -DADSTACK_MAX_SPACES=200000 in CFLAGS)"
  ifx --version 2>&1 | head -1 || true
  tapenade -version 2>&1 | head -3 || true
  java -version 2>&1 | head -1 || true
  echo "--- code/ ---"
  ls -l "$out/code"
  echo "--- code/ against the forward reference code/ ---"
  for f in SIZE.h CPP_OPTIONS.h EXF_OPTIONS.h OBCS_OPTIONS.h; do
    cmp "$RPAIR_CASE/code/$f" "$out/code/$f" && echo "same $f"
  done
  echo "--- forward reference packages.conf ---"
  cat "$RPAIR_CASE/code/packages.conf"
  echo "--- optfile against the reference optfile ---"
  diff "$RPAIR_OPTFILE" "$out/optfile" || true
} > "$out/identity.txt" 2>&1

# ---- 5. genmake2 / make --------------------------------------------------------
cd "$out/build"
set +e
"$RPAIR_MIT_ROOT/tools/genmake2" -rootdir "$RPAIR_MIT_ROOT" -mods "$out/code" \
    -of "$out/optfile" -mpi -tap > genmake.log 2>&1
rc=$?; echo "genmake2 rc=$rc"
[ $rc -eq 0 ] || { tail -30 genmake.log; exit 3; }
grep -q 'DADSTACK_MAX_SPACES=200000' Makefile \
  || { echo 'FATAL: ADSTACK_MAX_SPACES not in Makefile CFLAGS'; exit 3; }
make depend > depend.log 2>&1
rc=$?; echo "make depend rc=$rc"
[ $rc -eq 0 ] || { tail -30 depend.log; exit 4; }
# The build directory must compile the edited files, not the tree's.
for f in ini_pressure.F ini_fields.F initialise_varia.F cost_tile.F cost_test.F; do
  cmp "$out/code/$f" "$f" || { echo "FATAL: build/$f is not code/$f"; exit 4; }
done
make -j"$(nproc)" > forward_build.log 2>&1
rc=$?; echo "make forward rc=$rc"
[ -f mitgcmuv ] || { tail -40 forward_build.log; exit 5; }
make tap_adj > adjoint_build.log 2>&1
rc=$?; echo "make tap_adj rc=$rc"
set -e
grep -n "TAPENADE_FLAGS\|^AD_FILES\|-head" Makefile | head -5 > "$out/tapenade_head.txt" || true
if [ -f mitgcmuv_tap_adj ]; then
  ls -l mitgcmuv mitgcmuv_tap_adj
  echo "BUILD_COMPLETE_ADJ $out"
else
  echo "BUILD_FORWARD_ONLY $out (tap_adj failed, see adjoint_build.log)"
  tail -60 adjoint_build.log
  exit 6
fi
