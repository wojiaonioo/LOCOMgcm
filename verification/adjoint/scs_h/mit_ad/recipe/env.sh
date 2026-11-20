#!/usr/bin/env bash
# Source inside a compute allocation. Paths and the toolchain are configurable.
RPAIR_REPO=${RPAIR_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}
# Preserve the caller's model Python before activating the native build tools.
RPAIR_PY=${RPAIR_PY:-${DEFGCM_PYTHON:-$(command -v python)}}
source "${RPAIR_ENV_SETUP:-$RPAIR_REPO/verification/adjoint/cs32/paired/env.sh}"

# Forward reference tree at upstream MITgcm revision d861cd50.
RPAIR_MIT_ROOT=$(RPAIR_MIT_ROOT="${RPAIR_MIT_ROOT:-}" "$RPAIR_PY" -m defgcm.external_paths mitgcm_root) || { return 2 2>/dev/null || exit 2; }
RPAIR_OPTFILE=$(RPAIR_OPTFILE="${RPAIR_OPTFILE:-}" "$RPAIR_PY" -m defgcm.external_paths mitgcm_optfile) || { return 2 2>/dev/null || exit 2; }
# MIT forward reference: code/, input/, run/ of the explicit-shear pairing run.
RPAIR_CASE=${RPAIR_CASE:-$RPAIR_REPO/outputs/data/rpair-expl-20260928a/mitgcm}
# RPAIR_WORK selects the output directory.
RPAIR_WORK=${RPAIR_WORK:-$RPAIR_REPO/outputs/data/rpair_ad}
RPAIR_MPIRUN=${RPAIR_MPIRUN:-mpirun}
RPAIR_NPROCS=${RPAIR_NPROCS:-40}
export RPAIR_REPO RPAIR_MIT_ROOT RPAIR_OPTFILE RPAIR_CASE RPAIR_WORK RPAIR_MPIRUN RPAIR_NPROCS RPAIR_PY
