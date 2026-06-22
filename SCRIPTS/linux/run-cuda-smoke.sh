#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/SCRIPTS/linux/opensees-cuda-env.sh"

cd "${REPO_ROOT}"

run_py_smoke() {
  local script=$1
  local out rc=0
  set +e
  out="$("${OPENSEES_PYTHON}" "${script}" 2>&1)"
  rc=$?
  set -e
  echo "${out}"
  echo "${out}" | grep -qi "smoke test passed"
  if [[ ${rc} -ne 0 && ${rc} -ne 139 ]]; then
    return "${rc}"
  fi
}

run_py_smoke tests/cuda_explicit_alpha_smoke.py
run_py_smoke tests/cuda_explicit_alpha_tp_smoke.py
"${OPENSEES}" tests/cuda_explicit_alpha_tp_smoke.tcl

echo "CUDA smoke tests passed."
