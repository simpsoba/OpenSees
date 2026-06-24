# ShearBuilding40 — OpenFresco hybrid example (sequential, SP, MP)

40-story 1-D shear building with a local OpenFresco experimental element. Three Tcl scripts share the same model settings; only the executable and solver/partitioning differ.

## Contents

| File | Description |
|------|-------------|
| `ShearBuilding40.tcl` | Sequential reference (`OpenSeesFresco`, BandGeneral) |
| `ShearBuilding40SP.tcl` | MPI domain decomposition (`OpenSeesSPFresco`, Mumps) |
| `ShearBuilding40MP.tcl` | MPI manual partition (`OpenSeesMPFresco`, Mumps) |
| `elcentro.txt` | El Centro ground motion record |
| `plot_compare.py` | Story displacement plots → `compare_displacements_<mode>.png` |
| `bin/` | OpenSees Fresco executables and DLLs (see Runtime layout) |
| `lib/tcl8.6/` | Tcl 8.6 scripts for dynamic `tcl86t.dll` |

## Prerequisites

- Intel oneAPI with MPI (`mpiexec`) for SP/MP
- Python with `matplotlib` and `numpy` for plotting
- VS 2022 C++ redistributable

## Runtime layout

`bin\` must contain:

| File | Role |
|------|------|
| `OpenSeesFresco.exe` | Sequential hybrid simulation |
| `OpenSeesSPFresco.exe` | MPI domain decomposition |
| `OpenSeesMPFresco.exe` | MPI manual partition |
| `OpenFrescoTcl.dll` | OpenFresco Tcl interface |
| `tcl86t.dll` | Tcl runtime |
| `libcrypto-3-x64.dll`, `libssl-3-x64.dll` | OpenSSL (if linked dynamically) |

`lib\tcl8.6\` must contain the Tcl 8.6 script library used by `tcl86t.dll`.

## Run

From a plain `cmd.exe` shell in this folder (`conda deactivate` if MPI fails with `c0000135`):

```bat
cd EXAMPLES\ShearBuilding40SP

call "C:\Program Files (x86)\Intel\oneAPI\setVars.bat" intel64 mod
set PATH=%CD%\bin;%PATH%
set TCL_LIBRARY=%CD%\lib\tcl8.6
```

Sequential, SP (4 ranks), MP (4 ranks), then plot:

```bat
bin\OpenSeesFresco.exe ShearBuilding40.tcl
mpiexec -n 4 bin\OpenSeesSPFresco.exe ShearBuilding40SP.tcl
mpiexec -n 4 bin\OpenSeesMPFresco.exe ShearBuilding40MP.tcl
python plot_compare.py
```

Both `analytical` and `local` branches (6 runs + 2 plots):

```bat
bin\OpenSeesFresco.exe ShearBuilding40.tcl
mpiexec -n 4 bin\OpenSeesSPFresco.exe ShearBuilding40SP.tcl
mpiexec -n 4 bin\OpenSeesMPFresco.exe ShearBuilding40MP.tcl
python plot_compare.py --mode analytical

set SHEAR40_MODE=local
bin\OpenSeesFresco.exe ShearBuilding40.tcl
mpiexec -n 4 bin\OpenSeesSPFresco.exe ShearBuilding40SP.tcl
mpiexec -n 4 bin\OpenSeesMPFresco.exe ShearBuilding40MP.tcl
python plot_compare.py --mode local
```

SP only:

```bat
mpiexec -n 4 bin\OpenSeesSPFresco.exe ShearBuilding40SP.tcl
```

Adjust the `setVars.bat` path and `python` command to match your machine.

### MPI / conda note

If SP/MP fail with `c0000135` while sequential works, use a plain `cmd.exe` shell without conda on `PATH`, and ensure `bin\` contains all executables and DLLs listed above.

## Settings (top of each `.tcl` file)

- `expElementMode` — `analytical` or `local` (default: `analytical`; override with env `SHEAR40_MODE`)
- `ExpEleTag` — story with the experimental element (default 20)
- `analysisScheme` — `NewmarkExplicit`, `AlphaOSGeneralized`, or `Newmark`
- `zeta`, `w1`, `w2` — Rayleigh damping (default 5% at 2 and 20 rad/s)

## Output folders

Output folders always include the mode name (`output-<mode>`, `output-sp-<mode>`, `output-mp-<mode>`).

| Script | Default (`analytical`) | With `SHEAR40_MODE=local` |
|--------|------------------------|---------------------------|
| `ShearBuilding40.tcl` | `output-analytical/` | `output-local/` |
| `ShearBuilding40SP.tcl` | `output-sp-analytical/` | `output-sp-local/` |
| `ShearBuilding40MP.tcl` | `output-mp-analytical/` | `output-mp-local/` |

Each run deletes and recreates its output folder.

## Success markers

- Sequential / MP: `Finished transient analysis (3120 steps)`
- SP: `Partition done` then `Finished transient analysis (3120 steps)`
