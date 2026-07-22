#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# OpenSeesMP Ubuntu/WSL build with METIS 5 + CUDA (cuDSS / optional AmgX).
# Ninja + Conan + Intel MPI/MKL. Builds PARALLEL_PROCESSING=OFF (OpenSeesMP).
#
# Override paths via the environment, e.g.:
#   IMPI_ROOT=/opt/intel/oneapi/mpi/latest \
#   MUMPS_DIR=/path/to/mumps/build \
#   METIS5_DIR=/path/to/metis5/prefix \
#   CUDAToolkit_ROOT=/usr/local/cuda-12.4 \
#   ./makeOpenSeesMP_Ubuntu.sh
#
# CPU-only (no CUDA): OPS_SKIP_CUDA=1 ./makeOpenSeesMP_Ubuntu.sh
# ---------------------------------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-mp}"
MUMPS_DIR="${MUMPS_DIR:-${ROOT_DIR}/../mumps/build}"
METIS5_DIR="${METIS5_DIR:-/usr}"
IMPI_ROOT="${IMPI_ROOT:-/opt/intel/oneapi/mpi/2021.16}"
MKL_LIB="${MKL_LIB:-/opt/intel/oneapi/mkl/2025.2/lib}"
CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-/usr/local/cuda}"
AMGX_NO_MPI_DIR="${AMGX_NO_MPI_DIR:-}"
CUDA_ARCHS="${CMAKE_CUDA_ARCHITECTURES:-}"
OPS_SKIP_CUDA="${OPS_SKIP_CUDA:-0}"
JOBS="${JOBS:-$(nproc)}"

cd "${ROOT_DIR}"

if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
  set +u
  # shellcheck source=/dev/null
  source /opt/intel/oneapi/setvars.sh --force
  set -u
fi

if [[ ! -x "${IMPI_ROOT}/bin/mpigcc" ]]; then
  echo "ERROR: Intel MPI not found under: ${IMPI_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${MUMPS_DIR}/libdmumps.a" && ! -f "${MUMPS_DIR}/libdmumps.so" ]]; then
  echo "ERROR: MUMPS libraries not found under: ${MUMPS_DIR}" >&2
  exit 1
fi
if [[ ! -f "${METIS5_DIR}/include/metis.h" ]]; then
  echo "ERROR: METIS 5 headers not found under: ${METIS5_DIR}" >&2
  echo "Install libmetis-dev or set METIS5_DIR." >&2
  exit 1
fi

WITH_CUDA=0
if [[ "${OPS_SKIP_CUDA}" != "1" ]]; then
  if [[ ! -x "${CUDAToolkit_ROOT}/bin/nvcc" ]]; then
    echo "ERROR: CUDA not found at CUDAToolkit_ROOT=${CUDAToolkit_ROOT}" >&2
    echo "Set CUDAToolkit_ROOT, or OPS_SKIP_CUDA=1 for a CPU-only OpenSeesMP." >&2
    exit 1
  fi
  WITH_CUDA=1
  export CUDA_HOME="${CUDAToolkit_ROOT}"
  export PATH="${CUDAToolkit_ROOT}/bin${PATH:+:${PATH}}"
  export CUDACXX="${CUDAToolkit_ROOT}/bin/nvcc"
fi

# MUMPS was built against Intel MPI and LP64 MKL ScaLAPACK.
SCALAPACK_LIBRARIES="${SCALAPACK_LIBRARIES:-\
${MKL_LIB}/libmkl_scalapack_lp64.so;\
${MKL_LIB}/libmkl_gf_lp64.so;\
${MKL_LIB}/libmkl_gnu_thread.so;\
${MKL_LIB}/libmkl_core.so;\
${MKL_LIB}/libmkl_blacs_intelmpi_lp64.so}"

echo "BUILD_DIR=${BUILD_DIR}"
echo "IMPI_ROOT=${IMPI_ROOT}"
echo "MKL_LIB=${MKL_LIB}"
echo "MUMPS_DIR=${MUMPS_DIR}"
echo "METIS5_DIR=${METIS5_DIR}"
echo "WITH_CUDA=${WITH_CUDA}"
if [[ "${WITH_CUDA}" -eq 1 ]]; then
  echo "CUDAToolkit_ROOT=${CUDAToolkit_ROOT}"
  [[ -n "${AMGX_NO_MPI_DIR}" ]] && echo "AMGX_NO_MPI_DIR=${AMGX_NO_MPI_DIR}"
  [[ -n "${CUDA_ARCHS}" ]] && echo "CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}"
fi
echo "JOBS=${JOBS}"

# Conan provides Tcl, HDF5, Eigen and zlib.
conan install . -of "${BUILD_DIR}" \
  -s build_type=Release \
  -s arch=x86_64 \
  --build=missing \
  -c tools.cmake.cmaketoolchain:generator=Ninja

# Conan 2 layouts can place the toolchain in either location.
TOOLCHAIN="${BUILD_DIR}/build/Release/generators/conan_toolchain.cmake"
if [[ ! -f "${TOOLCHAIN}" ]]; then
  TOOLCHAIN="${BUILD_DIR}/Release/generators/conan_toolchain.cmake"
fi
if [[ ! -f "${TOOLCHAIN}" ]]; then
  echo "ERROR: Conan CMake toolchain was not generated." >&2
  exit 1
fi

CMAKE_ARGS=(
  -S .
  -B "${BUILD_DIR}/Release"
  -G Ninja
  -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}"
  -DCMAKE_BUILD_TYPE=Release
  -DMPI_C_COMPILER="${IMPI_ROOT}/bin/mpigcc"
  -DMPI_CXX_COMPILER="${IMPI_ROOT}/bin/mpigxx"
  -DMPI_Fortran_COMPILER="${IMPI_ROOT}/bin/mpif90"
  -DMUMPS_DIR="${MUMPS_DIR}"
  -DMETIS5_DIR="${METIS5_DIR}"
  -DSCALAPACK_LIBRARIES="${SCALAPACK_LIBRARIES}"
  -UOPENMPI
  -DPARALLEL_PROCESSING=OFF
)

if [[ "${WITH_CUDA}" -eq 1 ]]; then
  CMAKE_ARGS+=(
    -DCUDAToolkit_ROOT="${CUDAToolkit_ROOT}"
    -Ucudss_DIR
    -Ucudss_INCLUDE_DIR
    -Ucudss_LIBRARY_DIR
    -Ucudss_BINARY_DIR
  )
  if [[ -n "${AMGX_NO_MPI_DIR}" ]]; then
    CMAKE_ARGS+=(-DAMGX_NO_MPI_DIR="${AMGX_NO_MPI_DIR}")
  fi
  if [[ -n "${CUDA_ARCHS}" ]]; then
    CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}")
  fi
fi

cmake "${CMAKE_ARGS[@]}"

cmake --build "${BUILD_DIR}/Release" \
  --target OpenSeesMP \
  --parallel "${JOBS}"

# OpenSeesMP looks for Tcl under build-mp/lib at runtime.
if [[ -f "${ROOT_DIR}/SCRIPTS/linux/stage-tcl-runtime.sh" ]]; then
  bash "${ROOT_DIR}/SCRIPTS/linux/stage-tcl-runtime.sh" "${BUILD_DIR}"
else
  # Fallback: match conanfile.py tcl/8.6.11
  TCL_VERSION="8.6.11"
  TCL_STAGED=0
  for TCL_PKG_LIB in "${HOME}/.conan2/p/b"/tcl*/p/lib; do
    INIT_TCL="${TCL_PKG_LIB}/tcl8.6/init.tcl"
    if [[ -f "${INIT_TCL}" ]] && grep -q "package require -exact Tcl ${TCL_VERSION}" "${INIT_TCL}"; then
      mkdir -p "${BUILD_DIR}/lib"
      cp -a "${TCL_PKG_LIB}/tcl8.6" "${TCL_PKG_LIB}/tcl8" "${BUILD_DIR}/lib/"
      TCL_STAGED=1
      break
    fi
  done
  if [[ "${TCL_STAGED}" -ne 1 ]]; then
    echo "ERROR: Conan Tcl ${TCL_VERSION} runtime was not found under ${HOME}/.conan2/p/b." >&2
    exit 1
  fi
fi

echo
echo "OpenSeesMP built successfully:"
echo "  ${BUILD_DIR}/Release/OpenSeesMP"
if [[ "${WITH_CUDA}" -eq 1 ]]; then
  echo "  CUDA enabled (CuDSS / DistributedCuDSS when cuDSS is found)."
fi
