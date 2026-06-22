#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:-CudaKRAlpha}"
RHO="${2:-0.5}"
SCALE="${3:-3.0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/SCRIPTS/linux/opensees-cuda-env.sh"

cd "${REPO_ROOT}/EXAMPLES/KRAlphaExplicit/Two-Story_MRF"
echo "Running: OpenSees two_story_MRF.tcl ${METHOD} ${RHO} ${SCALE}"
"${OPENSEES}" two_story_MRF.tcl "${METHOD}" "${RHO}" "${SCALE}"

results="$(find results -name results.txt -print -quit 2>/dev/null || true)"
if [[ -z "${results}" ]] || ! grep -q "COMPLETED successfully" "${results}"; then
  echo "ERROR: MRF run did not complete successfully" >&2
  exit 1
fi
echo "MRF_TCL_OK: ${results}"
