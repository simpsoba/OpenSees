#!/usr/bin/env bash
# Build OpenSees + OpenSeesPy with CUDA and cuDSS on Ubuntu/WSL.
# Requires: Ninja, Intel oneAPI MPI/MKL, MUMPS, CUDA, libcudss-dev (>= 0.8).
#
# Override paths via environment variables before running, e.g.:
#   MUMPS_DIR=$HOME/mumps/build IMPI_ROOT=/opt/intel/oneapi/mpi/2021.16 ./makeUbuntu_CUDA.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

BUILD_DIR="${BUILD_DIR:-build/Release}"
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

echo "BUILD_DIR=${BUILD_DIR}"
echo "IMPI_ROOT=${IMPI_ROOT}"
echo "MKL_LIB=${MKL_LIB}"
echo "MUMPS_DIR=${MUMPS_DIR}"
echo "CUDAToolkit_ROOT=${CUDAToolkit_ROOT}"
echo "JOBS=${JOBS}"

cmake -S . -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${HOME}/bin" \
  -DMPI_C_COMPILER="${IMPI_ROOT}/bin/mpigcc" \
  -DMPI_CXX_COMPILER="${IMPI_ROOT}/bin/mpigxx" \
  -DMPI_Fortran_COMPILER="${IMPI_ROOT}/bin/mpif90" \
  -DMUMPS_DIR="${MUMPS_DIR}" \
  -DSCALAPACK_LIBRARIES="${SCALAPACK_LIBRARIES}" \
  -DCUDAToolkit_ROOT="${CUDAToolkit_ROOT}" \
  -Ucudss_DIR -Ucudss_INCLUDE_DIR -Ucudss_LIBRARY_DIR -Ucudss_BINARY_DIR

cmake --build "${BUILD_DIR}" --target OpenSees OpenSeesPy -j"${JOBS}"
cp -f "${BUILD_DIR}/OpenSeesPy.so" "${BUILD_DIR}/opensees.so"

echo
echo "Build complete."
echo "  ${BUILD_DIR}/OpenSees"
echo "  ${BUILD_DIR}/opensees.so"
echo "Run smokes: SCRIPTS/linux/run-cuda-smoke.sh"
