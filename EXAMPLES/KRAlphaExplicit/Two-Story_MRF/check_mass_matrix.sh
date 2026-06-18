#!/usr/bin/env bash
# Check assembled mass matrix for a Two-Story MRF mass mode (GimmeMCK + printA).
#
# Usage:
#   ./check_mass_matrix.sh              # modes 0, 1, 2
#   ./check_mass_matrix.sh 1            # mass mode 1 only
#   ./check_mass_matrix.sh 0 1          # modes 0 and 1
#
# Reports: mass_matrix_report_mode<N>.txt (+ dense/sparse matrix dumps alongside)

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

modes=("$@")
if [[ ${#modes[@]} -eq 0 ]]; then
  modes=(0 1 2)
fi

for m in "${modes[@]}"; do
  echo "=== massMode=${m} ==="
  "${OPENSEES}" two_story_MRF.tcl CHECK_MASS -massMode "${m}"
done
