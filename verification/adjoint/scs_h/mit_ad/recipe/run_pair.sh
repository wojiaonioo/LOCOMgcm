#!/usr/bin/env bash
# Run the regional forward and adjoint executables in one allocation.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
tag=${1:?usage: run_pair.sh TAG BUILD_DIR}
build=${2:?missing BUILD_DIR}
rc_all=0
for mode in fwd adj; do
  echo "=== $(date -Is) ${tag}_$mode"
  if bash "$here/run_mit.sh" "${tag}_$mode" "$mode" "$build"; then
    echo "--- ${tag}_$mode done $(date -Is)"
  else
    echo "--- ${tag}_$mode FAILED rc=$? $(date -Is)"; rc_all=1
  fi
done
echo "PAIR_DONE tag=$tag rc=$rc_all"
exit $rc_all
