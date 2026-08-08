#!/usr/bin/env bash
# OpenSeesMP sandbox driver for two_story_MRF_mp.tcl
#
# Builds the full MRF on every rank, auto-partitions with METIS, then runs:
#   MultiSOE × {Mumps, DistributedCuDSS}
#   Cuda*    × DistributedCuDSS
# Tip histories (tip_disp.out.*) are compared across backends.
#
# Usage:
#   ./run_mp_integrators.sh                  # quick matrix, np=2
#   ./run_mp_integrators.sh --full           # full ground-motion length
#   ./run_mp_integrators.sh --np 4 --rho 1.0
#   ./run_mp_integrators.sh --cases core     # MultiSOE KR + CudaKR only
#   ./run_mp_integrators.sh --cases all      # KR/MKR × midpoint/TP × systems

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# Prefer the dedicated OpenSeesMP tree; fall back to the shared CUDA build.
MP_BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-mp/Release}"
if [[ ! -x "${MP_BUILD_DIR}/OpenSeesMP" && -x "${REPO_ROOT}/build/Release/OpenSeesMP" ]]; then
  MP_BUILD_DIR="${REPO_ROOT}/build/Release"
fi
MP_TCL_LIBRARY=""
if [[ -d "${MP_BUILD_DIR%/Release}/lib/tcl8.6" ]]; then
  MP_TCL_LIBRARY="${MP_BUILD_DIR%/Release}/lib/tcl8.6"
elif [[ -d "${REPO_ROOT}/build-mp/lib/tcl8.6" ]]; then
  MP_TCL_LIBRARY="${REPO_ROOT}/build-mp/lib/tcl8.6"
fi
# shellcheck disable=SC1091
source "${REPO_ROOT}/SCRIPTS/linux/opensees-cuda-env.sh"
export BUILD_DIR="${MP_BUILD_DIR}"
if [[ -n "${MP_TCL_LIBRARY}" ]]; then
  export TCL_LIBRARY="${MP_TCL_LIBRARY}"
fi
export OPENSEESMP="${OPENSEESMP:-${BUILD_DIR}/OpenSeesMP}"
export MPIEXEC="${MPIEXEC:-${IMPI_ROOT:-/opt/intel/oneapi/mpi/2021.16}/bin/mpirun}"
NP=2
RHO=0.5
SCALE=3.0
QUICK=1
CASES=core
EXTRA_FLAGS=()
TOL=1.0e-6
CONSTRAINTS=Transformation

usage() {
  cat <<EOF
OpenSeesMP Two-Story MRF sandbox

  ./run_mp_integrators.sh [options]

Options:
  --np N           MPI ranks (default: 2)
  --rho R          spectral radius (default: 0.5)
  --scale S        ground-motion scale (default: 3.0)
  --full           full earthquake + free vibration (default is -quick)
  --quick          short ~1s transient (default)
  --cases core|all core = KR MultiSOE + CudaKR; all = KR/MKR × mid/TP
  --constraints H  Transformation (default) or Penalty
  --tol T          relative tip agreement tolerance (default: 1e-6)
  --openseesmp P   path to OpenSeesMP binary
  --               remaining flags forwarded to two_story_MRF_mp.tcl

OPENSEESMP=${OPENSEESMP}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --np) NP="$2"; shift 2 ;;
    --rho) RHO="$2"; shift 2 ;;
    --scale) SCALE="$2"; shift 2 ;;
    --full) QUICK=0; shift ;;
    --quick) QUICK=1; shift ;;
    --cases) CASES="$2"; shift 2 ;;
    --constraints) CONSTRAINTS="$2"; shift 2 ;;
    --tol) TOL="$2"; shift 2 ;;
    --openseesmp) OPENSEESMP="$2"; shift 2 ;;
    --) shift; EXTRA_FLAGS+=("$@"); break ;;
    *) EXTRA_FLAGS+=("$1"); shift ;;
  esac
done

if [[ "${CONSTRAINTS}" != "Transformation" && "${CONSTRAINTS}" != "Penalty" ]]; then
  echo "ERROR: --constraints must be Transformation or Penalty (got ${CONSTRAINTS})" >&2
  exit 1
fi

if [[ ! -x "${OPENSEESMP}" ]]; then
  echo "ERROR: OpenSeesMP not found/executable at ${OPENSEESMP}" >&2
  exit 1
fi
if [[ "${NP}" -lt 2 ]]; then
  echo "ERROR: need --np >= 2" >&2
  exit 1
fi

QUICK_FLAG=()
if [[ "${QUICK}" -eq 1 ]]; then
  QUICK_FLAG=(-quick)
fi

declare -a RUNS=()
add_run() {
  # args: label integrator system
  RUNS+=("$1|$2|$3")
}

case "${CASES}" in
  core)
    add_run "MultiSOE_KR_Mumps" KRAlphaExplicitMultiSOE Mumps
    add_run "MultiSOE_KR_DistCuDSS" KRAlphaExplicitMultiSOE DistributedCuDSS
    add_run "CudaKR_DistCuDSS" CudaKRAlpha DistributedCuDSS
    ;;
  all)
    for integ in KRAlphaExplicitMultiSOE MKRAlphaExplicitMultiSOE \
                 KRAlphaExplicitMultiSOE_TP MKRAlphaExplicitMultiSOE_TP; do
      add_run "${integ}_Mumps" "${integ}" Mumps
      add_run "${integ}_DistCuDSS" "${integ}" DistributedCuDSS
    done
    for integ in CudaKRAlpha CudaMKRAlpha CudaKRAlpha_TP CudaMKRAlpha_TP; do
      add_run "${integ}_DistCuDSS" "${integ}" DistributedCuDSS
    done
    ;;
  *)
    echo "ERROR: --cases must be core or all (got ${CASES})" >&2
    exit 1
    ;;
esac

echo "== OpenSeesMP MRF sandbox  np=${NP} rho=${RHO} scale=${SCALE} quick=${QUICK} cases=${CASES} constraints=${CONSTRAINTS}"
echo "   binary: ${OPENSEESMP}"

if [[ "${CONSTRAINTS}" == "Penalty" ]]; then
  SUMMARY_DIR="${SCRIPT_DIR}/results_mp/_compare_penalty"
else
  SUMMARY_DIR="${SCRIPT_DIR}/results_mp/_compare"
fi
mkdir -p "${SUMMARY_DIR}"
COMPARE_LIST=()

for entry in "${RUNS[@]}"; do
  IFS='|' read -r label integ system <<<"${entry}"
  echo
  echo "-- ${label}: integrator=${integ} system=${system} constraints=${CONSTRAINTS}"
  out_dir="${SUMMARY_DIR}/${label}"
  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"
  set +e
  "${MPIEXEC}" -np "${NP}" "${OPENSEESMP}" two_story_MRF_mp.tcl \
    "${integ}" "${RHO}" "${SCALE}" \
    -system "${system}" \
    -constraints "${CONSTRAINTS}" \
    -outdir "${out_dir}" \
    "${QUICK_FLAG[@]}" \
    "${EXTRA_FLAGS[@]}"
  mpirc=$?
  set -e
  if [[ ! -f "${out_dir}/results.txt" ]] || ! grep -q 'Analysis COMPLETED successfully' "${out_dir}/results.txt"; then
    echo "ERROR: analysis did not complete for ${label} (mpirun rc=${mpirc})" >&2
    exit 1
  fi
  hist="$(ls -1 "${out_dir}"/tip_disp.out.* 2>/dev/null | head -n 1 || true)"
  if [[ -z "${hist}" ]]; then
    echo "ERROR: missing tip_disp.out.* in ${out_dir}" >&2
    exit 1
  fi
  dest="${SUMMARY_DIR}/${label}.tip"
  cp -f "${hist}" "${dest}"
  COMPARE_LIST+=("${label}|${integ}:${dest}")
  echo "   captured tip history -> ${dest}"
done

python3 - "${TOL}" "${COMPARE_LIST[@]}" <<'PY'
import math, re, sys
from pathlib import Path
from collections import defaultdict

tol = float(sys.argv[1])

def family(integ: str) -> str:
    # Group MultiSOE and Cuda backends that implement the same algorithm.
    name = integ
    name = re.sub(r'^Cuda', '', name)
    name = name.replace('ExplicitMultiSOE', '').replace('Alpha', '')
    # e.g. KR, MKR, KR_TP, MKR_TP
    return name

groups = defaultdict(list)
for item in sys.argv[2:]:
    meta, path = item.split(":", 1)
    label, integ = meta.split("|", 1)
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        t, u = line.split()[:2]
        rows.append((float(t), float(u)))
    if not rows:
        raise SystemExit(f"ERROR: empty tip history for {label} ({path})")
    groups[family(integ)].append((label, rows))

ok = True
for fam in sorted(groups.keys()):
    entries = groups[fam]
    ref_label, ref = entries[0]
    print(f"\n== Tip agreement [{fam}] vs reference {ref_label} (tol={tol:g})")
    if len(entries) == 1:
        print(f"  SKIP {ref_label}: only one backend in this family")
        continue
    for label, rows in entries[1:]:
        n = min(len(ref), len(rows))
        if n == 0:
            print(f"FAIL {label}: no overlapping samples")
            ok = False
            continue
        num = 0.0
        den = 0.0
        max_abs = 0.0
        for i in range(n):
            du = rows[i][1] - ref[i][1]
            num += du * du
            den += ref[i][1] * ref[i][1]
            max_abs = max(max_abs, abs(du))
        rel = math.sqrt(num / den) if den > 0.0 else math.sqrt(num)
        status = "OK" if rel <= tol else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {status} {label}: rel_l2={rel:.3e} max_abs={max_abs:.3e} n={n}")

if not ok:
    raise SystemExit("ERROR: OpenSeesMP MultiSOE/Cuda tip histories disagree")
print("\nOpenSeesMP MultiSOE/Cuda sandbox tip comparison passed.")
PY

echo
echo "All requested OpenSeesMP sandbox cases finished."
