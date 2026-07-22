#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/SCRIPTS/linux/opensees-cuda-env.sh"

IMPI_ROOT="${IMPI_ROOT:-/opt/intel/oneapi/mpi/2021.16}"
OPENSEESMP="${OPENSEESMP:-${BUILD_DIR}/OpenSeesMP}"
NP="${NP:-2}"

if [[ ! -x "${OPENSEESMP}" ]]; then
  echo "ERROR: OpenSeesMP not found at ${OPENSEESMP}" >&2
  exit 1
fi

if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
  set +u
  # shellcheck source=/dev/null
  source /opt/intel/oneapi/setvars.sh --force >/dev/null
  set -u
fi

echo "=== truss MP smoke (np=2) ==="
"${IMPI_ROOT}/bin/mpirun" -np 2 "${OPENSEESMP}" \
  "${REPO_ROOT}/tests/distributed_cudss_mp_smoke.tcl"

echo "=== block3D static+transient MP regression (np=${NP}) ==="
"${IMPI_ROOT}/bin/mpirun" -np "${NP}" "${OPENSEESMP}" \
  "${REPO_ROOT}/tests/distributed_cudss_mp_block3d.tcl"

echo "DistributedCuDSS OpenSeesMP regression suite passed."
