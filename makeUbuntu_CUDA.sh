#!/usr/bin/env bash
# Build OpenSees, OpenSeesMP, OpenSeesSP, and OpenSeesPy with CUDA and cuDSS on Ubuntu/WSL.
#   build/Release      PARALLEL_PROCESSING=OFF  -> OpenSees, OpenSeesMP, OpenSeesPy
#   build-sp/Release   PARALLEL_PROCESSING=ON   -> OpenSeesSP
# Requires: Ninja, Intel oneAPI MPI/MKL, MUMPS, CUDA, libcudss-dev (>= 0.8).
#
# Override paths via environment variables before running, e.g.:
#   MUMPS_DIR=$HOME/mumps/build IMPI_ROOT=/opt/intel/oneapi/mpi/2021.16 ./makeUbuntu_CUDA.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

BUILD_DIR="${BUILD_DIR:-build/Release}"
BUILD_SP_DIR="${BUILD_SP_DIR:-build-sp/Release}"
IMPI_ROOT="${IMPI_ROOT:-/opt/intel/oneapi/mpi/2021.16}"
MKL_LIB="${MKL_LIB:-/opt/intel/oneapi/mkl/2025.2/lib}"
MUMPS_DIR="${MUMPS_DIR:-${HOME}/mumps/build}"
CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-/usr/local/cuda}"
JOBS="${JOBS:-$(nproc)}"

SCALAPACK_LIBRARIES="${SCALAPACK_LIBRARIES:-\
${MKL_LIB}/libmkl_scalapack_lp64.so;\
${MKL_LIB}/libmkl_gf_lp64.so;\
${MKL_LIB}/libmkl_gnu_thread.so;\
${MKL_LIB}/libmkl_core.so;\
${MKL_LIB}/libmkl_blacs_intelmpi_lp64.so}"

if [[ ! -x "${IMPI_ROOT}/bin/mpigcc" ]]; then
  echo "ERROR: Intel MPI not found at IMPI_ROOT=${IMPI_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${MUMPS_DIR}/libdmumps.a" && ! -f "${MUMPS_DIR}/libdmumps.so" ]]; then
  echo "ERROR: MUMPS not found at MUMPS_DIR=${MUMPS_DIR}" >&2
  exit 1
fi
if [[ ! -x "${CUDAToolkit_ROOT}/bin/nvcc" ]]; then
  echo "ERROR: CUDA not found at CUDAToolkit_ROOT=${CUDAToolkit_ROOT}" >&2
  exit 1
fi

BUILD_ROOT="$(dirname "${BUILD_DIR}")"
BUILD_SP_ROOT="$(dirname "${BUILD_SP_DIR}")"

echo "BUILD_DIR=${BUILD_DIR}"
echo "BUILD_SP_DIR=${BUILD_SP_DIR}"
echo "IMPI_ROOT=${IMPI_ROOT}"
echo "MKL_LIB=${MKL_LIB}"
echo "MUMPS_DIR=${MUMPS_DIR}"
echo "CUDAToolkit_ROOT=${CUDAToolkit_ROOT}"
echo "JOBS=${JOBS}"
echo "Build trees: ${BUILD_ROOT} (OpenSees, OpenSeesMP, OpenSeesPy)  ${BUILD_SP_ROOT} (OpenSeesSP)"

if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
  set +u
  # shellcheck source=/dev/null
  source /opt/intel/oneapi/setvars.sh --force
  set -u
fi

if ! command -v conan >/dev/null 2>&1; then
  echo "ERROR: conan not found. Install with: pip install conan" >&2
  exit 1
fi

configure_cuda_tree() {
  local cfg_build_dir="$1"
  local parallel_processing="$2"
  local parallel_flag="-DPARALLEL_PROCESSING=OFF"
  if [[ "${parallel_processing}" == "ON" ]]; then
    parallel_flag="-DPARALLEL_PROCESSING=ON"
  fi

  echo
  echo "=== Configuring ${cfg_build_dir} (PARALLEL_PROCESSING=${parallel_processing}) ==="

  conan install . -of "${cfg_build_dir}" --build=missing \
    -c tools.cmake.cmaketoolchain:generator=Ninja

  local cfg_toolchain="${cfg_build_dir}/build/generators/conan_toolchain.cmake"
  if [[ ! -f "${cfg_toolchain}" ]]; then
    cfg_toolchain="${cfg_build_dir}/generators/conan_toolchain.cmake"
  fi
  if [[ ! -f "${cfg_toolchain}" ]]; then
    echo "ERROR: Conan toolchain not found under ${cfg_build_dir}/build/generators or ${cfg_build_dir}/generators" >&2
    exit 1
  fi

  cmake -S . -B "${cfg_build_dir}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="${cfg_toolchain}" \
    -DCMAKE_INSTALL_PREFIX="${HOME}/bin" \
    -DMPI_C_COMPILER="${IMPI_ROOT}/bin/mpigcc" \
    -DMPI_CXX_COMPILER="${IMPI_ROOT}/bin/mpigxx" \
    -DMPI_Fortran_COMPILER="${IMPI_ROOT}/bin/mpif90" \
    -DMUMPS_DIR="${MUMPS_DIR}" \
    -DSCALAPACK_LIBRARIES="${SCALAPACK_LIBRARIES}" \
    -DCUDAToolkit_ROOT="${CUDAToolkit_ROOT}" \
    "${parallel_flag}" \
    -Ucudss_DIR -Ucudss_INCLUDE_DIR -Ucudss_LIBRARY_DIR -Ucudss_BINARY_DIR
}

configure_cuda_tree "${BUILD_DIR}" OFF
configure_cuda_tree "${BUILD_SP_DIR}" ON

cmake --build "${BUILD_DIR}" --target OpenSees OpenSeesMP -j"${JOBS}"
cmake --build "${BUILD_DIR}" --target OpenSeesPy -j"${JOBS}"
cmake --build "${BUILD_SP_DIR}" --target OpenSeesSP -j"${JOBS}"
cp -f "${BUILD_DIR}/OpenSeesPy.so" "${BUILD_DIR}/opensees.so"

# OpenSees looks for Tcl under <tree-root>/lib/tcl8.6; clock.tcl also needs lib/tcl8 (msgcat).
bash "${REPO_ROOT}/SCRIPTS/linux/stage-tcl-runtime.sh" "${BUILD_ROOT}" "${BUILD_SP_ROOT}"

echo
echo "Build complete."
echo "  ${BUILD_DIR}/OpenSees"
echo "  ${BUILD_DIR}/OpenSeesMP"
echo "  ${BUILD_SP_DIR}/OpenSeesSP"
echo "  ${BUILD_DIR}/opensees.so"
echo "Run smokes: SCRIPTS/linux/run-cuda-smoke.sh"
