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
- `CudaKRAlpha` / `CudaMKRAlpha` (GPU midpoint; requires CuDSS build)
- `CudaKRAlpha_TP` / `CudaMKRAlpha_TP` (GPU trapezoidal rule; requires CuDSS build)

### Contents
- `SDOF-OpenSees/`: single-DOF **plain transient** example (Python + `sdof.py`) — **serial only**.
- `Two-Story_MRF/`: two-story MRF model from Kolay & Ricles papers + ground motion file.
  - Serial: `two_story_MRF.tcl` / `two_story_MRF.py` / `run_integrators.py`
  - OpenSeesMP: `two_story_MRF_mp.tcl` + `run_mp_integrators.sh` (METIS `partition`, MultiSOE×Mumps/DistributedCuDSS, Cuda×DistributedCuDSS)

### Notes
- Serial SDOF / MRF drivers are **OpenSeesPy** (need `numpy` / `scipy` / etc., or local `build/Release/opensees.so`).
- OpenSeesMP MRF is **Tcl** (`two_story_MRF_mp.tcl`): every rank builds the full mesh, then METIS `partition`, gravity on Mumps, transient on the requested system.
- OpenSeesMP backend rules for this sandbox:
  - `*MultiSOE*` → `Mumps` or `DistributedCuDSS`
  - `Cuda*` → `DistributedCuDSS` only
- Tip histories are compared within each algorithm family (KR / MKR × midpoint / TP), not across families.
- Prefer Intel MPI `mpirun` and `build-mp/Release/OpenSeesMP` (see `run_mp_integrators.sh`).

### Ground motion file format
The example set includes both:
- **PEER header** files (`*.AT2`) — preferred, since `ReadRecord.py` can parse `dt/nPts`
  and generate a numeric `*.dat`.
- **numeric-only** `*.dat` files — kept for convenience, but not required.

### Quick runs
- SDOF (serial):
  - `python3 EXAMPLES/KRAlphaExplicit/SDOF-OpenSees/run_integrators.py`
- Two-Story MRF (serial):
  - `python3 EXAMPLES/KRAlphaExplicit/Two-Story_MRF/run_integrators.py`
- Two-Story MRF (OpenSeesMP, auto-partition):
  - `EXAMPLES/KRAlphaExplicit/Two-Story_MRF/run_mp_integrators.sh`                  # core: MultiSOE KR×{Mumps,DistCuDSS} + CudaKR
  - `EXAMPLES/KRAlphaExplicit/Two-Story_MRF/run_mp_integrators.sh --cases all --quick`
  - `EXAMPLES/KRAlphaExplicit/Two-Story_MRF/run_mp_integrators.sh --cases all --full`
  - Single case: `mpirun -np 2 OpenSeesMP two_story_MRF_mp.tcl KRAlphaExplicitMultiSOE 0.5 3.0 -system Mumps -quick`

### Using the locally built Python module
The example Python scripts prefer importing the locally built module at:
`build/Release/opensees.so` (imported as `import opensees as ops`).
If it is not available, they fall back to `openseespy.opensees`.

