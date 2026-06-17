#!/usr/bin/env bash

conda activate py312-gpu
source /opt/intel/oneapi/setvars.sh

export MKL_LIB=/opt/intel/oneapi/mkl/2025.2/lib
export LD_LIBRARY_PATH="$MKL_LIB:${LD_LIBRARY_PATH:-}"
export OPENSEES=/home/garaujor/OpenSees-CUDA/build/Release/OpenSees

# Two-Story MRF
cd /home/garaujor/OpenSees-CUDA/EXAMPLES/KRAlphaExplicit/Two-Story_MRF
python3 run_integrators.py --engine tcl --jobs 1
python3 run_integrators.py --cudss-dffi --append --engine tcl --jobs 1

# SDOF (after Two-Story finishes)
cd /home/garaujor/OpenSees-CUDA/EXAMPLES/KRAlphaExplicit/SDOF-OpenSees
python3 run_integrators.py --jobs 1
python3 run_integrators.py --cudss-dffi --append --jobs 1
