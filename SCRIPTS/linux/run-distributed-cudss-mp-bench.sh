#!/usr/bin/env bash
# Sweep block3D sizes: correctness + serial vs MP analyze timings + history CSVs.
#
# Defaults:
#   - small mesh with UmfPack+CuDSS (serial) for correctness
#   - denser DOF sweep up to ~10k (CuDSS / Mumps / DistributedCuDSS)
#   - no ProfileSPD
#   - short transient (0.5 s / 50 steps)
#
# Usage:
#   SCRIPTS/linux/run-distributed-cudss-mp-bench.sh
#   SIZES="2 2 4;4 4 8;6 6 16;8 8 24;8 8 32;8 8 64" NPS="1 2 4 8" MODE=both SCRIPTS/linux/run-distributed-cudss-mp-bench.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_ROOT}/SCRIPTS/linux/opensees-cuda-env.sh"

IMPI_ROOT="${IMPI_ROOT:-/opt/intel/oneapi/mpi/2021.16}"
OPENSEES="${OPENSEES:-${BUILD_DIR}/OpenSees}"
OPENSEESMP="${OPENSEESMP:-${BUILD_DIR}/OpenSeesMP}"
MODE="${MODE:-both}"
# Free DOFs ≈ (nx+1)*(ny+1)*nz*3; nz must be divisible by NPS (1/2/4/8).
#   2 2 4  => 108   (skipped for np=8)
#   4 4 8  => 600
#   6 6 16 => 2352
#   8 8 24 => 5832
#   8 8 32 => 7776
#   8 8 64 => 15552
SIZES="${SIZES:-2 2 4;4 4 8;6 6 16;8 8 24;8 8 32;8 8 64}"
NPS="${NPS:-1 2 4 8}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/tests/out}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="${OUT_DIR}/block3d_bench_${STAMP}.log"
CSV="${OUT_DIR}/block3d_bench_${STAMP}.csv"
HIST_CSV="${OUT_DIR}/block3d_bench_${STAMP}_history.csv"

mkdir -p "${OUT_DIR}"

if [[ ! -x "${OPENSEES}" || ! -x "${OPENSEESMP}" ]]; then
  echo "ERROR: need OpenSees and OpenSeesMP under ${BUILD_DIR}" >&2
  exit 1
fi

if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
  set +u
  # shellcheck source=/dev/null
  source /opt/intel/oneapi/setvars.sh --force >/dev/null
  set -u
fi

MPIRUN="${IMPI_ROOT}/bin/mpirun"
TCL="${REPO_ROOT}/tests/distributed_cudss_mp_block3d_bench.tcl"

echo "Logging to ${LOG}"
echo "CSV to ${CSV}"
echo "HISTORY CSV to ${HIST_CSV}"
echo "mode=${MODE} sizes='${SIZES}' nps='${NPS}'" | tee "${LOG}"

echo "type,np,nx,ny,nz,nodes,eles,ndof,system,numberer,analyze_ms,tip_norm,ux,uy,uz" > "${CSV}"
echo "type,np,nx,ny,nz,ndof,system,numberer,step,time,ux,uy,uz,tip_norm" > "${HIST_CSV}"

run_one() {
  local np=$1 nx=$2 ny=$3 nz=$4
  local out
  echo
  echo "=== np=${np} mesh=${nx}x${ny}x${nz} mode=${MODE} ===" | tee -a "${LOG}"
  if [[ "${np}" -eq 1 ]]; then
    out="$("${OPENSEES}" "${TCL}" "${nx}" "${ny}" "${nz}" "${MODE}" 2>&1)" || {
      echo "${out}" | tee -a "${LOG}"
      echo "ERROR: serial run failed" >&2
      exit 1
    }
  else
    out="$("${MPIRUN}" -np "${np}" "${OPENSEESMP}" "${TCL}" "${nx}" "${ny}" "${nz}" "${MODE}" 2>&1)" || {
      echo "${out}" | tee -a "${LOG}"
      echo "ERROR: MP run failed" >&2
      exit 1
    }
  fi
  echo "${out}" | tee -a "${LOG}"

  echo "${out}" | awk '
    /^BENCH / {
      typ=""; npv=""; nx=""; ny=""; nz=""; nodes=""; eles=""; ndof=""; sysname=""; numberer=""; ms=""; tip=""; tipn=""
      n=split($0, a, " ")
      for (i=1;i<=n;i++) {
        split(a[i], kv, "=")
        key=kv[1]; val=substr(a[i], index(a[i],"=")+1)
        if (key=="type") typ=val
        else if (key=="np") npv=val
        else if (key=="nx") nx=val
        else if (key=="ny") ny=val
        else if (key=="nz") nz=val
        else if (key=="nodes") nodes=val
        else if (key=="eles") eles=val
        else if (key=="ndof") ndof=val
        else if (key=="system") sysname=val
        else if (key=="numberer") numberer=val
        else if (key=="analyze_ms") ms=val
        else if (key=="tip") tip=val
        else if (key=="tip_norm") tipn=val
      }
      if (tip=="" || tip=="NA") next
      split(tip, t, ",")
      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", \
        typ,npv,nx,ny,nz,nodes,eles,ndof,sysname,numberer,ms,tipn,t[1],t[2],t[3]
    }
  ' >> "${CSV}"

  echo "${out}" | awk '
    /^HISTORY / {
      typ=""; npv=""; nx=""; ny=""; nz=""; ndof=""; sysname=""; numberer=""; step=""; t=""; ux=""; uy=""; uz=""; tipn=""
      n=split($0, a, " ")
      for (i=1;i<=n;i++) {
        split(a[i], kv, "=")
        key=kv[1]; val=substr(a[i], index(a[i],"=")+1)
        if (key=="type") typ=val
        else if (key=="np") npv=val
        else if (key=="nx") nx=val
        else if (key=="ny") ny=val
        else if (key=="nz") nz=val
        else if (key=="ndof") ndof=val
        else if (key=="system") sysname=val
        else if (key=="numberer") numberer=val
        else if (key=="step") step=val
        else if (key=="time") t=val
        else if (key=="ux") ux=val
        else if (key=="uy") uy=val
        else if (key=="uz") uz=val
        else if (key=="tip_norm") tipn=val
      }
      if (typ=="") next
      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", \
        typ,npv,nx,ny,nz,ndof,sysname,numberer,step,t,ux,uy,uz,tipn
    }
  ' >> "${HIST_CSV}"
}

IFS=';' read -r -a SIZE_ARR <<< "${SIZES}"
read -r -a NP_ARR <<< "${NPS}"

for size in "${SIZE_ARR[@]}"; do
  size="$(echo "${size}" | xargs)"
  [[ -z "${size}" ]] && continue
  read -r nx ny nz <<< "${size}"
  for np in "${NP_ARR[@]}"; do
    if [[ "${np}" -gt 1 && $((nz % np)) -ne 0 ]]; then
      echo "SKIP np=${np} nz=${nz} (nz not divisible by np)" | tee -a "${LOG}"
      continue
    fi
    run_one "${np}" "${nx}" "${ny}" "${nz}"
  done
done

python3 - "${CSV}" <<'PY' | tee -a "${LOG}"
import csv, sys
from collections import defaultdict

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
if not rows:
    print("No BENCH rows parsed.")
    sys.exit(0)

print("\n=== Correctness (max |tip_norm_i - tip_norm_j| within each np/mesh/type) ===")
groups = defaultdict(list)
for r in rows:
    key = (r["type"], r["np"], r["nx"], r["ny"], r["nz"])
    groups[key].append(r)

ok = True
for key, items in sorted(groups.items(), key=lambda x: (x[0][0], int(x[0][1]), int(x[0][2]), int(x[0][3]), int(x[0][4]))):
    norms = [float(i["tip_norm"]) for i in items]
    spread = max(norms) - min(norms)
    systems = ",".join(i["system"] for i in items)
    print(f"type={key[0]} np={key[1]} mesh={key[2]}x{key[3]}x{key[4]} systems={systems} tip_norm_spread={spread:.6e}")
    if spread > 1e-6:
        ok = False
        print("  FAIL spread exceeds 1e-6")

tips = {}
for r in rows:
    tips[(r["type"], r["np"], r["nx"], r["ny"], r["nz"], r["system"])] = (
        float(r["tip_norm"]), float(r["ux"]), float(r["uy"]), float(r["uz"])
    )

print("\n=== Cross-np CUDA tip agreement (serial CuDSS vs DistributedCuDSS) ===")
meshes = sorted({(r["type"], r["nx"], r["ny"], r["nz"]) for r in rows},
                key=lambda t: (t[0], int(t[1]), int(t[2]), int(t[3])))
nps = sorted({int(r["np"]) for r in rows})
for typ, nx, ny, nz in meshes:
    serial = tips.get((typ, "1", nx, ny, nz, "CuDSS"))
    if serial is None:
        continue
    for n in nps:
        if n == 1:
            continue
        mp = tips.get((typ, str(n), nx, ny, nz, "DistributedCuDSS"))
        if mp is None:
            continue
        dxyz = max(abs(serial[i] - mp[i]) for i in range(1, 4))
        print(f"type={typ} mesh={nx}x{ny}x{nz} np=1->{n} max|du|={dxyz:.6e}")
        if dxyz > 1e-6:
            ok = False
            print("  FAIL serial CuDSS vs DistributedCuDSS")

pair = [
    ("CuDSS", "DistributedCuDSS"),
    ("UmfPack", "UmfPack"),
    ("Mumps", "Mumps"),
]
by_ms = {}
for r in rows:
    by_ms[(r["type"], str(r["np"]), r["nx"], r["ny"], r["nz"], r["system"])] = int(r["analyze_ms"])

print("\n=== Timing summary (analyze_ms) ===")
header = f"{'type':<10} {'mesh':<10} {'system':<28} " + " ".join(f"{'np'+str(n):>8}" for n in nps) + f"{'sp_npmax':>10}"
print(header)
print("-" * len(header))

for typ, nx, ny, nz in meshes:
    mesh = f"{nx}x{ny}x{nz}"
    for serial, parallel in pair:
        times = {}
        any_data = False
        for n in nps:
            sysname = serial if n == 1 else parallel
            if n == 1 and serial in ("Mumps", "ProfileSPD"):
                times[n] = None
                continue
            if n > 1 and parallel == "UmfPack":
                times[n] = None
                continue
            if n == 1 and serial == "ProfileSPD":
                times[n] = None
                continue
            ms = by_ms.get((typ, str(n), nx, ny, nz, sysname))
            times[n] = ms
            if ms is not None:
                any_data = True
        if not any_data:
            continue
        t1 = times.get(1)
        sp = ""
        max_np = max((n for n in nps if times.get(n) is not None), default=None)
        if t1 and max_np and max_np > 1 and times.get(max_np):
            sp = f"{t1 / times[max_np]:.2f}x"
        cols = " ".join(f"{(times[n] if times.get(n) is not None else '-'):>8}" for n in nps)
        label = serial if serial == parallel else f"{serial}/{parallel}"
        print(f"{typ:<10} {mesh:<10} {label:<28} {cols}{sp:>10}")

if not ok:
    sys.exit(1)
print("\nBench correctness checks passed.")
print(f"BENCH_CSV={path}")
PY

# Record paths for the plotter
echo "${CSV}" > "${OUT_DIR}/block3d_bench_latest.txt"
echo "${HIST_CSV}" >> "${OUT_DIR}/block3d_bench_latest.txt"

echo
echo "Done."
echo "  log:  ${LOG}"
echo "  csv:  ${CSV}"
echo "  hist: ${HIST_CSV}"
