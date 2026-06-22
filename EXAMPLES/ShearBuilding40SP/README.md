# ShearBuilding40 — OpenFresco hybrid example (sequential, SP, MP)

40-story 1-D shear building with a local OpenFresco experimental element. Three Tcl scripts share the same model settings; only the executable and solver/partitioning differ.

## Contents

| File | Description |
|------|-------------|
| `ShearBuilding40.tcl` | Sequential reference (`OpenSeesFresco`, BandGeneral) |
| `ShearBuilding40SP.tcl` | MPI domain decomposition (`OpenSeesSPFresco`, Mumps) |
| `ShearBuilding40MP.tcl` | MPI manual partition (`OpenSeesMPFresco`, Mumps) |
| `elcentro.txt` | El Centro ground motion record |
| `run_compare.bat` | Run all three cases and plot displacements |
| `plot_compare.py` | Story displacement plots → `compare_displacements.png` (full record) and `compare_displacements_freevib.png` (t ≥ 31.2 s) |
| `run.bat` | SP-only launcher (`mpiexec -n 4`) |
| `bin/` | Fresco executables + runtime DLLs (optional; falls back to `build/bin-fresco/`) |
| `lib/tcl8.6/` | Tcl 8.6 scripts for dynamic `tcl86t.dll` |

## Prerequisites

- Intel oneAPI with MPI (`mpiexec`) for SP/MP
- Python with `matplotlib` and `numpy` for plotting
- VS 2022 C++ redistributable

## Run

All three cases + comparison plot:

```bat
run_compare.bat
```

SP only:

```bat
run.bat
```

## Settings (top of each `.tcl` file)

- `expElementMode` — `twoNodeLink` or `generic` (default)
- `ExpEleTag` — story with the experimental element (default 20)
- `analysisScheme` — `NewmarkExplicit`, `AlphaOSGeneralized`, or `Newmark`
- `zeta`, `w1`, `w2` — Rayleigh damping (default 5% at 2 and 20 rad/s)

## Output folders

| Script | Output folder |
|--------|---------------|
| `ShearBuilding40.tcl` | `output/` |
| `ShearBuilding40SP.tcl` | `output-sp/` |
| `ShearBuilding40MP.tcl` | `output-mp/` |

Each folder holds `node_*_disp.out`, `Elmt_Frc.out`, `Elmt_ctrlDsp.out`, and a run log. Folders are cleared at the start of each run.

## Success markers

- Sequential / MP: `Finished transient analysis (3120 steps)`
- SP: `Partition done` then `Finished transient analysis (3120 steps)`
