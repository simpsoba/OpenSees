#!/usr/bin/env bash
# Shared runtime env for OpenSees-CUDA on Linux/WSL.
# Source from other scripts: source SCRIPTS/linux/opensees-cuda-env.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build/Release}"

IMPI_ROOT="${IMPI_ROOT:-/opt/intel/oneapi/mpi/2021.16}"
MKL_LIB="${MKL_LIB:-/opt/intel/oneapi/mkl/2025.2/lib}"
CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-/usr/local/cuda}"

CUDSS_LIB="${CUDSS_LIB:-/usr/lib/x86_64-linux-gnu/libcudss/12}"

export PYTHONPATH="${BUILD_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${MKL_LIB}:${IMPI_ROOT}/lib/release:${IMPI_ROOT}/lib:${CUDSS_LIB}:${CUDAToolkit_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export OPENSEES="${BUILD_DIR}/OpenSees"
export OPENSEES_PYTHON="${OPENSEES_PYTHON:-python3}"
