#!/usr/bin/env bash
# Manual single-run cookbook for two_story_MRF.tcl
#
# Usage:
#   ./run_single.sh              # print examples
#   ./run_single.sh ARGS...      # run: $OPENSEES two_story_MRF.tcl ARGS...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -z "${OPENSEES:-}" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
  export OPENSEES="${REPO_ROOT}/build/Release/OpenSees"
fi

if [[ -d /opt/intel/oneapi/mkl/2025.2/lib ]]; then
  export MKL_LIB=/opt/intel/oneapi/mkl/2025.2/lib
  export LD_LIBRARY_PATH="${MKL_LIB}:${LD_LIBRARY_PATH:-}"
fi

if [[ $# -eq 0 ]]; then
  cat <<EOF
Two-Story MRF — single transient run via Tcl

  \$OPENSEES two_story_MRF.tcl <integrator> [rho] [scale] [flags...]

Integrators:
  Newmark | NewmarkCPU | KRAlphaExplicit | MKRAlphaExplicit
  KRAlphaExplicitMultiSOE | MKRAlphaExplicitMultiSOE | CudaKRAlpha | CudaMKRAlpha

Flags (after optional rho / scale):
  -massMode 0|1|2          0=consistent, 1=element lumped, 2=nodal lumped
  -numberer Plain|RCM|AMD
  -incrementalAccel        MultiSOE / CUDA only
  -alphaCloseCheck         MultiSOE / CUDA only
  -system UmfPack|SuperLU|FullGeneral|CuDSS
  -cudssPrecision dFFI
  -cudssIrNSteps N

Newmark note: γ=0.5 and β=0.25 are fixed inside the Tcl script (not passed on argv).
  ./run_single.sh Newmark
  ./run_single.sh Newmark -system UmfPack -massMode 1 -numberer Plain
  ./run_single.sh NewmarkCPU -massMode 0

KR / MKR (dense CPU, ρ on argv):
  ./run_single.sh KRAlphaExplicit 1.0
  ./run_single.sh MKRAlphaExplicit 1.0 3.0 -massMode 0 -numberer RCM
  ./run_single.sh KRAlphaExplicit 1.0 3.0 -numberer Plain

MultiSOE (CuDSS default; override with -system):
  ./run_single.sh KRAlphaExplicitMultiSOE 1.0 3.0
  ./run_single.sh MKRAlphaExplicitMultiSOE 1.0 3.0 -system SuperLU -massMode 1
  ./run_single.sh KRAlphaExplicitMultiSOE 1.0 3.0 -incrementalAccel -alphaCloseCheck

CUDA:
  ./run_single.sh CudaKRAlpha 1.0 3.0 -massMode 1
  ./run_single.sh CudaMKRAlpha 1.0 3.0 -incrementalAccel
  ./run_single.sh CudaKRAlpha 1.0 3.0 -cudssPrecision dFFI -cudssIrNSteps 2

OPENSEES=${OPENSEES}
EOF
  exit 0
fi

exec "${OPENSEES}" two_story_MRF.tcl "$@"
