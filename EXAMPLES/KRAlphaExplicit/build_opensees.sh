#!/usr/bin/env bash

conda activate py312-gpu
source /opt/intel/oneapi/setvars.sh

cd /home/garaujor/OpenSees-CUDA

IMPI_ROOT=/opt/intel/oneapi/mpi/2021.16
MKL_LIB=/opt/intel/oneapi/mkl/2025.2/lib
export LD_LIBRARY_PATH="$MKL_LIB:${LD_LIBRARY_PATH:-}"

# Uncomment for a clean rebuild (removes cmake cache and all build artifacts):
# rm -rf build/Release

cmake -S . -B build/Release \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/bin" \
  -DMPI_C_COMPILER="$IMPI_ROOT/bin/mpigcc" \
  -DMPI_CXX_COMPILER="$IMPI_ROOT/bin/mpigxx" \
  -DMPI_Fortran_COMPILER="$IMPI_ROOT/bin/mpif90" \
  -DMUMPS_DIR="/home/garaujor/mumps/build" \
  -DSCALAPACK_LIBRARIES="$MKL_LIB/libmkl_scalapack_lp64.so;$MKL_LIB/libmkl_gf_lp64.so;$MKL_LIB/libmkl_gnu_thread.so;$MKL_LIB/libmkl_core.so;$MKL_LIB/libmkl_blacs_intelmpi_lp64.so" \
  -DCUDAToolkit_ROOT=/usr/local/cuda

echo "Building OpenSees..."
cmake --build build/Release --target OpenSees -j"$(nproc)"

echo "Building OpenSeesPy..."
cmake --build build/Release --target OpenSeesPy -j"$(nproc)"

cp build/Release/OpenSeesPy.so build/Release/opensees.so
echo "Copied OpenSeesPy.so -> build/Release/opensees.so"
