## KRAlphaExplicit / MKRAlphaExplicit example set

This folder mirrors the example set you provided in:
`/home/garaujor/testing_KRAlphaExplicit_TP`.

It is intended as a **reproducible sandbox** for exercising:
- `KRAlphaExplicit`
- `KRAlphaExplicit_TP`
- `MKRAlphaExplicit`
- `MKRAlphaExplicit_TP`
- `KRAlphaExplicitMultiSOE` / `MKRAlphaExplicitMultiSOE`
- `KRAlphaExplicitMultiSOE_TP` / `MKRAlphaExplicitMultiSOE_TP`

### Contents
- `SDOF-OpenSees/`: single-DOF **plain transient** example (Python + `sdof.py`).
- `Two-Story_MRF/`: two-story MRF model from Kolay & Ricles papers + ground motion file.

### Notes
- These are **OpenSeesPy** scripts (Python), not Tcl scripts.
- To run them you need an environment with `openseespy` (and for the MRF script: `numpy`, `scipy`, `pandas`, `matplotlib`, `opsvis`).
- The integrator method names in OpenSeesPy match the Tcl names (e.g. `ops.integrator("KRAlphaExplicit_TP", rhoInf)`).
- `*MultiSOE` integrators require optional flags first (any order), then trailing workspace blocks in fixed order: `Begin_Mass_System {...} End_Mass_System` then `Begin_Alpha_System {...} End_Alpha_System`, e.g.
  `ops.integrator("KRAlphaExplicitMultiSOE", rhoInf, "-incrementalAccel", "Begin_Mass_System", "UmfPack", "End_Mass_System", "Begin_Alpha_System", "UmfPack", "End_Alpha_System")`.
  The example scripts append the two blocks at the end when the method name contains `MultiSOE`.

### Ground motion file format
The example set includes both:
- **PEER header** files (`*.AT2`) — preferred, since `ReadRecord.py` can parse `dt/nPts`
  and generate a numeric `*.dat`.
- **numeric-only** `*.dat` files — kept for convenience, but not required.

### Quick runs
- SDOF:
  - `python3 EXAMPLES/KRAlphaExplicit/SDOF-OpenSees/run_integrators.py`
- Two-Story MRF:
  - `python3 EXAMPLES/KRAlphaExplicit/Two-Story_MRF/run_integrators.py`

### Using the locally built Python module
The example Python scripts prefer importing the locally built module at:
`build/Release/opensees.so` (imported as `import opensees as ops`).
If it is not available, they fall back to `openseespy.opensees`.

