#!/usr/bin/env bash
# rm -fr build
#   build/Release      PARALLEL_PROCESSING=OFF  -> OpenSees, OpenSeesPy  (OpenSeesMP commented)
#   build-sp/Release   PARALLEL_PROCESSING=ON   -> OpenSeesSP (commented)
#
# CMake configure matches makeUbuntu_CUDA.sh (MPI, MUMPS, MKL ScaLAPACK), minus CUDA/cuDSS.
# Requires Intel oneAPI MPI + MUMPS on WSL — same as makeUbuntu_CUDA.sh.
# Override: MUMPS_DIR=$HOME/mumps/build IMPI_ROOT=/opt/intel/oneapi/mpi/2021.16 ./makeUbuntu.sh

set -e

BUILD_DIR=build/Release
BUILD_SP_DIR=build-sp/Release
IMPI_ROOT=${IMPI_ROOT:-/opt/intel/oneapi/mpi/2021.16}
MKL_LIB=${MKL_LIB:-/opt/intel/oneapi/mkl/2025.2/lib}
MUMPS_DIR=${MUMPS_DIR:-${HOME}/mumps/build}
JOBS=$(nproc)

echo "=== makeUbuntu.sh ==="
echo "BUILD_DIR=${BUILD_DIR}  BUILD_SP_DIR=${BUILD_SP_DIR}  JOBS=${JOBS}"

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
echo "Prerequisites OK (Intel MPI, MUMPS)"

if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
  set +u
  # shellcheck source=/dev/null
  source /opt/intel/oneapi/setvars.sh --force
  set -u
  echo "Sourced Intel oneAPI setvars.sh"
fi

find_toolchain() {
  local root="$1" t
  for t in \
    "${root}/build/Release/generators/conan_toolchain.cmake" \
    "${root}/build/generators/conan_toolchain.cmake" \
    "${root}/generators/conan_toolchain.cmake"; do
    [[ -f "${t}" ]] && { echo "${t}"; return 0; }
  done
  echo "ERROR: Conan toolchain not found under ${root}" >&2
  return 1
}

configure_tree() {
  local dir="$1" parallel="$2"
  local root toolchain
  root="$(dirname "${dir}")"
  echo
  echo "=== Configuring ${dir} (PARALLEL_PROCESSING=${parallel}) ==="
  conan install . -of "${root}" --build=missing \
    -c tools.cmake.cmaketoolchain:generator=Ninja
  echo "Conan install succeeded for ${root}"
  toolchain="$(find_toolchain "${root}")"
  echo "Using toolchain: ${toolchain}"
  cmake -S . -B "${dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="${toolchain}" \
    -DCMAKE_INSTALL_PREFIX="${HOME}/bin" \
    -DMPI_C_COMPILER="${IMPI_ROOT}/bin/mpigcc" \
    -DMPI_CXX_COMPILER="${IMPI_ROOT}/bin/mpigxx" \
    -DMPI_Fortran_COMPILER="${IMPI_ROOT}/bin/mpif90" \
    -DMUMPS_DIR="${MUMPS_DIR}" \
    -DSCALAPACK_LIBRARIES="${SCALAPACK_LIBRARIES}" \
    -DPARALLEL_PROCESSING="${parallel}"
  echo "CMake configure succeeded for ${dir}"
}

configure_tree "${BUILD_DIR}" OFF
configure_tree "${BUILD_SP_DIR}" ON

echo
echo "=== Building OpenSees ==="
cmake --build "${BUILD_DIR}" --target OpenSees -j"${JOBS}"
echo "OpenSees built successfully: ${BUILD_DIR}/OpenSees"

# cmake --build "${BUILD_DIR}" --target OpenSeesMP -j"${JOBS}"
# echo "OpenSeesMP built successfully: ${BUILD_DIR}/OpenSeesMP"

echo
echo "=== Building OpenSeesPy ==="
cmake --build "${BUILD_DIR}" --target OpenSeesPy -j"${JOBS}"
echo "OpenSeesPy built successfully: ${BUILD_DIR}/OpenSeesPy.so"

cp -f "${BUILD_DIR}/OpenSeesPy.so" "${BUILD_DIR}/opensees.so"
echo "Copied OpenSeesPy.so -> ${BUILD_DIR}/opensees.so"

# cmake --build "${BUILD_SP_DIR}" --target OpenSeesSP -j"${JOBS}"
# echo "OpenSeesSP built successfully: ${BUILD_SP_DIR}/OpenSeesSP"

echo
echo "=== Staging Tcl runtime ==="
for root in build build-sp; do
  for tcl_pkg in "${HOME}/.conan2/p/b"/tcl*/p/lib; do
    if [[ -f "${tcl_pkg}/tcl8.6/init.tcl" ]]; then
      mkdir -p "${root}/lib"
      cp -a "${tcl_pkg}/tcl8.6" "${tcl_pkg}/tcl8" "${root}/lib/"
      echo "Tcl copied successfully to ${root}/lib/"
      break
    fi
  done
done

echo
echo "=== Build complete ==="
echo "  ${BUILD_DIR}/OpenSees"
echo "  ${BUILD_DIR}/opensees.so"
echo "Run: cd ${BUILD_DIR} && ./OpenSees /mnt/c/.../YourModel.tcl"
