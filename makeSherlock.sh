#!/usr/bin/env bash
#
# makeSherlock.sh — build OpenSees (Tcl) and OpenSeesPy on Stanford Sherlock
#
# What this builds:
#   build/Release/OpenSees       sequential Tcl interpreter
#   build/Release/opensees.so    Python module (copy of OpenSeesPy.so)
#
# Requirements:
#   - Run from the root of this repository
#   - Run on a compute/dev node (sh_dev), not the login node, for long builds
#   - Sherlock modules: gcc, imkl/2019, python, cmake, ninja
#
# Usage:
#   chmod +x makeSherlock.sh
#   ./makeSherlock.sh
#
# First run creates a local Python virtual environment (.venv) and installs Conan.
# Re-run anytime to reconfigure/rebuild.
#
# After a successful build:
#   export TCL_LIBRARY=$PWD/build/lib/tcl8.6
#   ./build/Release/OpenSees yourModel.tcl
#
# Optional: MKL_LIB=/path/to/mkl/lib/intel64  JOBS=8 ./makeSherlock.sh

set -euo pipefail

cd "$(dirname "$0")"

# --- Sherlock modules -------------------------------------------------------
module purge
module load gcc/12.4.0
module load imkl/2019
module load python/3.12.1 cmake/3.31.4 ninja/1.13.1

export CC=gcc CXX=g++ FC=gfortran

for cmd in gcc g++ gfortran ninja cmake python3; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: ${cmd} not found after module load." >&2
    exit 1
  fi
done

# --- MKL LAPACK (sequential) ------------------------------------------------
# imkl/2019 sets MKLROOT; override with MKL_LIB=... if needed.
: "${MKLROOT:?module load imkl/2019 did not set MKLROOT}"
MKL_LIB="${MKL_LIB:-${MKLROOT}/lib/intel64}"
LAPACK_LIBRARIES="-Wl,--start-group;${MKL_LIB}/libmkl_intel_lp64.a;${MKL_LIB}/libmkl_sequential.a;${MKL_LIB}/libmkl_core.a;-Wl,--end-group;-lpthread"
echo "Using MKL LAPACK from ${MKL_LIB}"

# --- Python virtual environment (Conan) -------------------------------------
# Conan fetches build dependencies (Tcl, HDF5, Eigen, etc.). We keep it in .venv
# so nothing is installed into your user site-packages.
# First we check if the virtual environment exists, if not we create it
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -q conan
  .venv/bin/conan profile detect --force
fi
# Then we activate the virtual environment
source .venv/bin/activate

# --- Dependencies (Conan) ---------------------------------------------------
conan install . -of build --build=missing -c tools.cmake.cmaketoolchain:generator=Ninja

# --- Configure (CMake + Ninja) ----------------------------------------------
# PARALLEL_PROCESSING=OFF builds the sequential OpenSees target (not OpenSeesSP/MP).
cmake -S . -B build/Release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$(find build -name conan_toolchain.cmake | head -1)" \
  -DPARALLEL_PROCESSING=OFF \
  -DLAPACK_LIBRARIES="${LAPACK_LIBRARIES}"

# --- Compile ----------------------------------------------------------------
# Set the number of jobs to use for the build
JOBS="${JOBS:-$(nproc)}"
# Then we compile the OpenSees and OpenSeesPy targets
cmake --build build/Release --target OpenSees OpenSeesPy -j"${JOBS}"
# Then we copy and rename the OpenSeesPy.so file to opensees.so
cp -f build/Release/OpenSeesPy.so build/Release/opensees.so

# --- Tcl runtime scripts ----------------------------------------------------
# OpenSees looks for Tcl scripts next to the executable (../lib/tcl8.6).
# Must match conanfile.py (tcl/8.6.11).
TCL_VERSION=8.6.11
TCL_PKG_LIB=""
for d in "${HOME}/.conan2/p/b"/tcl*; do
  if [[ -f "${d}/p/lib/tcl8.6/init.tcl" ]] && \
     grep -q "package require -exact Tcl ${TCL_VERSION}" "${d}/p/lib/tcl8.6/init.tcl"; then
    TCL_PKG_LIB="${d}/p/lib"
    break
  fi
done
if [[ -z "${TCL_PKG_LIB}" ]]; then
  echo "ERROR: Conan Tcl ${TCL_VERSION} not found under ${HOME}/.conan2/p/b" >&2
  exit 1
fi
mkdir -p build/lib
cp -a "${TCL_PKG_LIB}/tcl8.6" "${TCL_PKG_LIB}/tcl8" build/lib/

# --- Done -------------------------------------------------------------------
echo "Done."
echo "  build/Release/OpenSees"
echo "  build/Release/opensees.so"
echo "Run: ./build/Release/OpenSees model.tcl"
