#!/usr/bin/env bash

conda activate py312-gpu
source /opt/intel/oneapi/setvars.sh

export MKL_LIB=/opt/intel/oneapi/mkl/2025.2/lib
export LD_LIBRARY_PATH="$MKL_LIB:${LD_LIBRARY_PATH:-}"
export OPENSEES=/home/garaujor/OpenSees-CUDA/build/Release/OpenSees

# --- Two-Story MRF sweep controls (edit here) ---
# Space-separated ρ values after --rho (e.g. "0.5 0.9 1.0" or "1.0").
RHOS="1.0"
# Mass models: --all-mass-modes (0=consistent, 1=element lumped, 2=nodal lumped)
#              or --mass-mode N for a single mode.
MASS_FLAGS="--mass-mode 0"
REUSE_NEWMARK="--reuse-newmark"

# Set to 1 to remove all results/figures (and timing CSVs) before the sweep.
# Pass 1 of run_integrators.py also wipes without --append; this clears everything
# up front (all mass-mode trees, SDOF, stale figure folders from prior runs).
WIPE_RESULTS=1

KRAlphaExplicit=/home/garaujor/OpenSees-CUDA/EXAMPLES/KRAlphaExplicit
TwoStory="${KRAlphaExplicit}/Two-Story_MRF"
SDOF="${KRAlphaExplicit}/SDOF-OpenSees"

if [[ "${WIPE_RESULTS}" == 1 ]]; then
  echo "Wiping prior results and figures..."
  for ex in "${TwoStory}" "${SDOF}"; do
    for sub in results results_1 results_2 figures figures_1 figures_2; do
      d="${ex}/${sub}"
      if [[ -d "${d}" ]]; then
        rm -rf "${d}"
        echo "  removed ${d}"
      fi
    done
    csv="${ex}/timing_summary.csv"
    if [[ -f "${csv}" ]]; then
      rm -f "${csv}"
      echo "  removed ${csv}"
    fi
  done
fi

# Two-Story MRF: default RCM numberer; standard + incremental integrators
cd "${TwoStory}"
python3 run_integrators.py --engine tcl --rho $RHOS $MASS_FLAGS
python3 run_integrators.py --alpha-close-check-only --append ${REUSE_NEWMARK} --engine tcl --rho 1.0 $MASS_FLAGS
python3 run_integrators.py --cudss-dffi-only --append ${REUSE_NEWMARK} --engine tcl --rho $RHOS $MASS_FLAGS
python3 run_integrators.py --cudss-dffi-only --alpha-close-check-only --append ${REUSE_NEWMARK} --engine tcl --rho 1.0 $MASS_FLAGS
# Append full matrix with Plain / AMD DOF numberers (results tagged *_Plain_* / *_AMD_*).
python3 run_integrators.py --numberer Plain --append ${REUSE_NEWMARK} --engine tcl --rho $RHOS $MASS_FLAGS
python3 run_integrators.py --numberer AMD --append ${REUSE_NEWMARK} --engine tcl --rho $RHOS $MASS_FLAGS

# SDOF (after Two-Story finishes)
cd "${SDOF}"
python3 run_integrators.py --rho $RHOS
python3 run_integrators.py --alpha-close-check-only --append --rho 1.0
python3 run_integrators.py --cudss-dffi-only --append --rho $RHOS
