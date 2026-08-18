# Windows build guide — OpenSees + OpenFresco (RTHS-CUDA)

Build **OpenSees** (CUDA optional, dynamic `tcl86t`, `loadPackage` ABI) and the
**OpenFresco** plugin (`Release-CMake`) on Windows from source. Run **`OpenSees.exe`**
and **`OpenSeesMP.exe`** for hybrid simulation — no separate Fresco-branded executable names.

Everything clones under workspace folder **`RTHS-CUDA`**. Conan runs in conda env
**`RTHS-CUDA`**.

### Quick links

* [Installation (paths and layout)](#installation)
* [Prerequisites](#prerequisites)
* [Building — prepare shell](#building--prepare-the-build-shell)
* [Building — clone repos](#building--clone-repositories)
* [Building — MUMPS](#building--mumps)
* [Building — METIS 5](#building--metis-5)
* [Building — OpenSees and OpenSeesMP (CPU-only)](#building--opensees-and-openseesmp-cpu-only)
* [Building — OpenSees and OpenSeesMP (with CUDA)](#building--opensees-and-openseesmp-with-cuda)
* [Building — stage OpenFresco libraries](#building--stage-openfresco-libraries)
* [Building — OpenFresco plugin](#building--openfresco-plugin)
* [Verify installation](#verify-installation)
* [Built executables and usage](#built-executables-and-usage)
* [Intel MPI runtime (Windows, single node)](#intel-mpi-runtime-windows-single-node)
* [OpenSeesMP — Tcl runtime after build](#openseesmp--tcl-runtime-after-build)
* [Troubleshooting](#troubleshooting)
* [Appendix: full copy-paste script](#appendix-full-copy-paste-script)

---

## Installation

Edit the variables below once, then work through **Prerequisites** and **Building**
in order. After each build step, run the matching **Verify** command to confirm success.

### Paths and variables

```text
# --- Workspace: everything clones under this folder ---
PARENT=C:\Users\YOURNAME\source\repos\simpsoba\RTHS-CUDA

# --- Git branches (both repos) ---
BRANCH=RTHS-CUDA

# --- Git remotes ---
OPENSEES_REPO=https://github.com/simpsoba/OpenSees.git
OPENFRESCO_REPO=https://github.com/simpsoba/OpenFresco.git

# --- Folder names under PARENT (OpenFresco MSBuild expects this OpenSees folder name) ---
OPENSEES_DIR=OpenSees
OPENFRESCO_DIR=OpenFresco

# --- Installed tools (change if your paths differ) ---
ONEAPI_SETVARS=C:\Program Files (x86)\Intel\oneAPI\setvars.bat
TCL_ROOT=C:\Program Files\Tcl
OPENSSL_ROOT=C:\Program Files\OpenSSL
CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9

# --- Conda env for Conan (name must match: RTHS-CUDA) ---
CONDA_ROOT=C:\Users\YOURNAME\AppData\Local\anaconda3
CONDA_ENV=%CONDA_ROOT%\envs\RTHS-CUDA

# --- MUMPS / METIS (sibling folders under PARENT; see layout below) ---
MUMPS_DIR=%PARENT%\mumps\build
METIS5_DIR=%PARENT%\metis-5.1.0\install

# --- OpenSees build (see Building — OpenSees and OpenSeesMP) ---
# CPU-only: omit -DCUDAToolkit_ROOT=... from cmake configure
# With CUDA: pass -DCUDAToolkit_ROOT=%CUDA_ROOT% on both build trees
JOBS=10               # parallel compile jobs
```

### Required folder layout

```text
C:\Users\YOURNAME\source\repos\simpsoba\RTHS-CUDA\   <-- PARENT (workspace root)
  OpenSees\                                        <- OpenSees repo (this guide lives here)
  OpenFresco\                                      <- OpenFresco repo
  mumps\                                           <- clone OpenSees/mumps
  mumps\build\                                     <- MUMPS_DIR (dmumps.lib + headers)
  metis-5.1.0\                                     <- METIS 5 sources
  metis-5.1.0\install\                             <- METIS5_DIR (optional, for partition())
```

---

## Prerequisites

Install on the new machine (versions should match your working box when possible):

| Component | Notes |
|-----------|--------|
| **Git** | https://git-scm.com/ |
| **Visual Studio 2022** | Desktop development with C++ |
| **Intel oneAPI Base + HPC Toolkits** | Fortran (`ifx`), MKL, MPI — `setvars.bat` at `ONEAPI_SETVARS` |
| **ActiveTcl 8.6** | Must provide `tcl86t.lib` and `tcl86t.dll` under `TCL_ROOT`. For **OpenSeesMP**, the DLL patch level must match the Tcl **script** tree you point `TCL_LIBRARY` at (see [OpenSeesMP — Tcl runtime](#openseesmp--tcl-runtime-after-build)) |
| **OpenSSL 3 (Win64, /MT)** | Headers: `%OPENSSL_ROOT%\include\openssl\ssl.h` |
| | Static libs: `%OPENSSL_ROOT%\lib\VC\x64\MT\libssl.lib`, `libcrypto.lib` |
| **Conan 2.x** | Inside conda env **`RTHS-CUDA`** — see **Create the conda env** below |
| **CMake + Ninja** | Bundled with VS 2022; add to `PATH` in the [build shell](#building--prepare-the-build-shell) |
| **MUMPS** | Build from source — [Building — MUMPS](#building--mumps) |
| **METIS 5.1.0** | Build from source — [Building — METIS 5](#building--metis-5) |
| **CUDA 12.x** (optional) | Required only for [Building — OpenSees and OpenSeesMP (with CUDA)](#building--opensees-and-openseesmp-with-cuda) |

### Create the conda env (Conan)

One-time setup. The environment name **`RTHS-CUDA`** matches the workspace folder name.

```bat
conda create -n RTHS-CUDA python=3.11 -y
conda activate RTHS-CUDA
pip install conan
conan --version
```

If Anaconda/Miniconda lives elsewhere, adjust `CONDA_ROOT` in the Configuration block
(e.g. `C:\Users\YOURNAME\anaconda3` instead of `%LOCALAPPDATA%\anaconda3`).

### Push branches from your dev machine (first time only)

The `RTHS-CUDA` branches must exist on GitHub before the new PC can clone them:

```bat
cd C:\path\to\OpenSees
git push -u origin RTHS-CUDA

cd C:\path\to\OpenFresco
git push -u origin RTHS-CUDA
```

Recovery tags (optional): `ops-cuda-pre-RTHS-CUDA`, `ops-fresco-pre-RTHS-CUDA`, `devel-pre-RTHS-CUDA`.

---

## Building

### Prepare the build shell

Use **cmd.exe** or **x64 Native Tools Command Prompt for VS 2022**.

```bat
set PARENT=C:\Users\YOURNAME\source\repos\simpsoba\RTHS-CUDA
mkdir "%PARENT%"
cd /d "%PARENT%"
```

Load oneAPI (needed for OpenSees, OpenFresco, and MUMPS):

```bat
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 mod
```

Activate the **`RTHS-CUDA`** conda env (provides Conan on PATH):

```bat
set CONDA_ROOT=C:\Users\YOURNAME\AppData\Local\anaconda3
set CONDA_ENV=%CONDA_ROOT%\envs\RTHS-CUDA
set PATH=%CONDA_ENV%;%CONDA_ENV%\Scripts;%PATH%
```

Or, if `conda activate` works in your shell:

```bat
conda activate RTHS-CUDA
```

Verify tools:

```bat
git --version
cmake --version
ninja --version
conan --version
ifx /help
where tcl86t.dll
```

**Verify:** toolchain is ready:

```bat
git --version && cmake --version && ninja --version && conan --version && ifx /help >nul && echo OK
```

---

### Clone repositories

```bat
cd /d "%PARENT%"

git clone -b RTHS-CUDA %OPENSEES_REPO% %OPENSEES_DIR%
git clone -b RTHS-CUDA %OPENFRESCO_REPO% %OPENFRESCO_DIR%

cd %OPENSEES_DIR%
git log -1 --oneline

cd ..\%OPENFRESCO_DIR%
git log -1 --oneline
```

If `git clone -b RTHS-CUDA` fails, the branch is not on the remote yet — complete the
**Push branches** step under Prerequisites first.

**Verify:**

```bat
dir "%PARENT%\OpenSees\CMakeLists.txt"
dir "%PARENT%\OpenFresco\WIN64\OpenFresco.sln"
```

---

### MUMPS

OpenSeesMP (and OpenSeesSP with parallel MUMPS solvers) links against a **pre-built MUMPS**
tree passed as `-DMUMPS_DIR=...`. On Windows, `MUMPS_DIR` must be the CMake **build**
directory (not a stripped `install\` prefix), because OpenSees also compiles against headers
under `MUMPS_DIR\_deps\mumps-src\include`.

Default layout:

```text
%PARENT%\mumps\build\dmumps.lib
%PARENT%\mumps\build\mumps_common.lib
%PARENT%\mumps\build\pord.lib
%PARENT%\mumps\build\_deps\mumps-src\include\dmumps_c.h
```

Build MUMPS **before** OpenSees. Pass `-DMUMPS_DIR=%PARENT%\mumps\build` when configuring
OpenSees (see below).

#### Build

OpenSees uses static `/MT` and double-precision MUMPS (`dmumps`). Match that here.
Load Intel oneAPI first (provides `ifx`, `icx`, MKL, MPI, and ScaLAPACK).

```bat
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 mod

cd /d "%PARENT%"
git clone https://github.com/OpenSees/mumps.git mumps

cd mumps
if not exist build mkdir build
cd build

cmake .. -G Ninja ^
  -Darith=d ^
  -Dopenmp=OFF ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
  -DBUILD_TESTING=OFF ^
  -DCMAKE_Fortran_COMPILER=ifx ^
  -DCMAKE_C_COMPILER=icx ^
  -DCMAKE_C_FLAGS="/DWIN32 /D_WINDOWS /Qiopenmp" ^
  -DCMAKE_Fortran_FLAGS="/nologo /fpp /Qiopenmp" ^
  -DCMAKE_EXE_LINKER_FLAGS="/Qiopenmp" ^
  -DCMAKE_C_FLAGS_RELEASE="/O3 /Ob2 /DNDEBUG" ^
  -DCMAKE_Fortran_FLAGS_RELEASE="/O3 /DNDEBUG" ^
  -DCMAKE_BUILD_TYPE=Release

cmake --build . --config Release --parallel 8
```

`-Darith=d` builds **double-precision only** (`dmumps.lib`), which is what OpenSees links.
Parallel MPI MUMPS is the default (`parallel=ON`); oneAPI MKL/MPI satisfy LAPACK and ScaLAPACK.

This copies the **WSL MUMPS recipe**: CMake `openmp=OFF`, and OpenMP comes in through compiler/linker
**FLAGS** (`-fopenmp` on Linux; `/Qiopenmp` is the ifx/icx equivalent). Do **not** also pass
`-Dopenmp=ON` — that is a second, different embedding (`find_package(OpenMP)` on the MUMPS targets)
and is not how the fast WSL tree was built. `-fPIC` is Linux-only; omit it on Windows.

CMake **Release** on Windows `icx`/`ifx` defaults to **`/O2`** (MSVC-style). Linux Release defaults to
**`-O3`**, which is what the fast WSL MUMPS tree used. Pass `CMAKE_*_FLAGS_RELEASE=/O3` on Windows
so the two trees match.

Confirm:

```bat
dir "%PARENT%\mumps\build\dmumps.lib"
dir "%PARENT%\mumps\build\mumps_common.lib"
dir "%PARENT%\mumps\build\pord.lib"
dir "%PARENT%\mumps\build\_deps\mumps-src\include\dmumps_c.h"
```

**Verify:** `dir "%PARENT%\mumps\build\dmumps.lib"` should succeed.

#### How OpenSees uses MUMPS

Pass the MUMPS **build** directory to CMake:

```bat
set MUMPS_DIR=%PARENT%\mumps\build
```

CMake flag: `-DMUMPS_DIR="%MUMPS_DIR%"`.

OpenSeesMP/SP on Windows then **link** Intel MKL ScaLAPACK with **`mkl_intel_thread` + `libiomp5md`**
(same role as WSL `mkl_gnu_thread` + `libgomp`). That is link-time only: OpenSees C++ is **not**
compiled with OpenMP (the WSL OpenSeesMP tree also had no `-fopenmp` on C/C++). Mixing two OpenMP
runtimes (MSVC `vcomp` vs Intel `libiomp5`) is the usual conflict — do not add MSVC `/openmp`
to OpenSees. Do not mix `mkl_sequential` with `mkl_intel_thread` on the same MPI executable.
At **run time**, pin MPI ranks and set `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` so those extra
MKL/MUMPS OpenMP threads do not oversubscribe the cores — see
[Intel MPI runtime](#intel-mpi-runtime-windows-single-node).

If you copy a MUMPS tree from another machine, copy the **entire** `mumps\build` folder
(including `_deps\`), not just the three `.lib` files.

#### Optional: smoke-test the MUMPS build

Reconfigure with tests enabled and run the bundled check executables:

```bat
cd /d "%PARENT%\mumps\build"
cmake .. -DBUILD_TESTING=ON
cmake --build . --config Release --target mumpscfg d_simple --parallel 4
ctest -C Release
```

---

### METIS 5

OpenSeesMP's Tcl `partition` command requires **METIS 5** headers and library.
The copy under `OTHER/METIS` in the OpenSees tree is **METIS 4** (legacy graph code only).
If METIS 5 is unavailable, OpenSeesMP still builds, but CMake warns that `partition`
will remain a stub.

There is no official Windows METIS binary. Build METIS 5.1.0 from source and install
to `%PARENT%\metis-5.1.0\install` (pass `-DMETIS5_DIR=...` when configuring OpenSeesMP).

#### Obtain METIS 5.1.0

Use the classic 5.1.0 release with bundled GKlib. This fork includes Windows build support:

```bat
cd /d "%PARENT%"
git clone --depth 1 https://github.com/eric2003/METIS-5.1.0-Modified.git metis-5.1.0
cd metis-5.1.0
```

**Do not** substitute the KarypisLab `master` branch (METIS 5.2.x). OpenSeesMP's
`partition` implementation targets the METIS 5.1 API, and `METIS_PartMeshNodal` from
5.2.x has caused access violations with MSVC.

#### Build with Visual Studio 2022

OpenSees uses the static MSVC runtime (`/MT`). Build METIS the same way. Keep 32-bit
`idx_t` (`METIS_ENABLE_64BIT=OFF`) because OpenSeesMP communicates these arrays using
`MPI_INT`.

From the `metis-5.1.0` directory:

```bat
set CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set PREFIX=%CD%\install

"%CMAKE%" -S . -B build -G "Visual Studio 17 2022" -A x64 ^
  -DMETIS_ENABLE_64BIT=OFF ^
  -DSHARED=FALSE ^
  -DCMAKE_INSTALL_PREFIX="%PREFIX%" ^
  -DCMAKE_POLICY_DEFAULT_CMP0091=NEW ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded

"%CMAKE%" --build build --config Release --target metis --parallel 8
```

Only the `metis` library target is needed. Assemble the installation prefix manually:

```bat
mkdir install\include install\lib
copy include\metis.h install\include\
copy build\libmetis\Release\metis.lib install\lib\
```

Expected layout:

```text
%PARENT%\metis-5.1.0\install\include\metis.h
%PARENT%\metis-5.1.0\install\lib\metis.lib
```

GKlib is included in `metis.lib`; it does not require a separate link step.

Confirm:

```bat
dir "%PARENT%\metis-5.1.0\install\include\metis.h"
dir "%PARENT%\metis-5.1.0\install\lib\metis.lib"
```

**Verify:** `dir "%PARENT%\metis-5.1.0\install\lib\metis.lib"` should succeed.

#### How OpenSeesMP uses METIS

```bat
set METIS5_DIR=%PARENT%\metis-5.1.0\install
```

CMake flags: `-DMETIS5_DIR="%METIS5_DIR%" -UOPENSEES_METIS5_LIBRARY`.
OpenSees (serial) does not require METIS; OpenSeesMP `partition()` does.

#### Confirm METIS was detected (OpenSees configure)

A successful CMake configuration includes messages similar to:

```text
-- METIS 5 (partition / OpenSeesMP): ... HAVE_METIS5_HEADER=TRUE
-- MPI executables: linking METIS 5 at .../lib/metis.lib
```

If configuration instead reports:

```text
METIS 5 headers not found; OpenSeesMP partition() will remain a stub
```

check `METIS5_DIR`, `include/metis.h`, and `lib/metis.lib`. If the prefix changed after an
earlier configure, delete the build directory or re-run with `-UOPENSEES_METIS5_LIBRARY`.

#### Quick partition checks (optional)

From `EXAMPLES\Partition`, use a smaller mesh for a fast MPI check:

```bat
cd /d "%PARENT%\%OPENSEES_DIR%\EXAMPLES\Partition"
..\..\build\Release\OpenSees.exe Example.tcl serial 4 4 20
mpiexec -n 4 ..\..\build-mp\Release\OpenSeesMP.exe Example.tcl metis 4 4 20
mpiexec -n 4 ..\..\build-mp\Release\OpenSeesMP.exe Example.tcl custom 4 4 20
mpiexec -n 4 ..\..\build-mp\Release\OpenSeesMP.exe Example.tcl samePart 4 4 20
```

Gravity displacement, eigenvalues, and transient displacement should agree between serial,
METIS, custom, and `samePart` runs up to normal floating-point roundoff.

Larger brick-column comparison (`EXAMPLES\LargeMP`):

```bat
cd /d "%PARENT%\%OPENSEES_DIR%\EXAMPLES\LargeMP"
..\..\build\Release\OpenSees.exe Example.tcl 4 4 20
mpiexec -n 4 ..\..\build-mp\Release\OpenSeesMP.exe Example.tcl 4 4 20
python compare_tip.py
```

Build **OpenSees.exe** (`build\Release`) and **OpenSeesMP.exe** (`build-mp\Release`) with Conan,
CMake, and Ninja. Pick CPU-only **or** CUDA configure flags below — not both on the same tree
without reconfiguring.

Shared environment for all OpenSees steps (run once per cmd session):

```bat
cd /d "%PARENT%\OpenSees"

call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 mod

set CONDA_ROOT=C:\Users\YOURNAME\AppData\Local\anaconda3
set CONDA_ENV=%CONDA_ROOT%\envs\RTHS-CUDA
set PATH=%CONDA_ENV%;%CONDA_ENV%\Scripts;%PATH%

set "CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"

set MUMPS_DIR=%PARENT%\mumps\build
set METIS5_DIR=%PARENT%\metis-5.1.0\install
set JOBS=10
```

OpenSees and OpenSeesMP use **separate** Conan/CMake trees: `build\Release` and
`build-mp\Release`. Both link dynamic **`tcl86t.dll`** on Windows (for OpenFresco
`loadPackage`).

**Reconfigure on Windows:** If `build-mp\Release\CMakeCache.txt` was created on Linux/WSL
(paths like `/home/...`), delete `build-mp\Release` and configure again on Windows.

---

### OpenSees and OpenSeesMP (CPU-only)

No CUDA toolkit required. Do **not** pass `-DCUDAToolkit_ROOT=...`.

**Important:** CMake still enables CUDA if `nvcc` is on `PATH`, even when you omit
`-DCUDAToolkit_ROOT`. For a CPU-only tree, temporarily remove `%CUDA_ROOT%\bin` from `PATH`
before configuring, or accept a CUDA-enabled binary (still fine for MPI/MUMPS runs that do
not call GPU solvers).

#### 1. Conan dependencies

```bat
cd /d "%PARENT%\OpenSees"

conan install . -of build -s build_type=Release -s arch=x86_64 -s compiler.runtime=static --build=missing -c tools.cmake.cmaketoolchain:generator=Ninja

conan install . -of build-mp -s build_type=Release -s arch=x86_64 -s compiler.runtime=static --build=missing -c tools.cmake.cmaketoolchain:generator=Ninja
```

Set the Conan toolchain path (Conan 2 may use either layout):

```bat
set TOOLCHAIN_OP=%CD%\build\Release\generators\conan_toolchain.cmake
if not exist "%TOOLCHAIN_OP%" set TOOLCHAIN_OP=%CD%\build\build\Release\generators\conan_toolchain.cmake

set TOOLCHAIN_MP=%CD%\build-mp\Release\generators\conan_toolchain.cmake
if not exist "%TOOLCHAIN_MP%" set TOOLCHAIN_MP=%CD%\build-mp\build\Release\generators\conan_toolchain.cmake
```

#### 2. Configure and build OpenSees.exe

```bat
"%CMAKE%" -S . -B build\Release -G Ninja ^
  -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN_OP%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
  -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DCMAKE_Fortran_COMPILER=ifx ^
  -DBLA_STATIC=ON ^
  -DMKL_LINK=static ^
  -DMKL_INTERFACE_FULL=intel_lp64 ^
  -DMUMPS_DIR="%MUMPS_DIR%" ^
  -DMETIS5_DIR="%METIS5_DIR%" ^
  -UOPENSEES_METIS5_LIBRARY ^
  -UOPENMPI ^
  -DPARALLEL_PROCESSING=OFF ^
  -DCMAKE_NINJA_FORCE_RESPONSE_FILE=ON ^
  -DCMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS=ON ^
  -DCMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS=ON ^
  -DCMAKE_Fortran_USE_RESPONSE_FILE_FOR_OBJECTS=ON

"%CMAKE%" --build build\Release --target OpenSees --parallel %JOBS%
```

#### 3. Configure and build OpenSeesMP.exe

METIS 5 must exist before this step (`%METIS5_DIR%\include\metis.h`).

```bat
"%CMAKE%" -S . -B build-mp\Release -G Ninja ^
  -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN_MP%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
  -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DCMAKE_Fortran_COMPILER=ifx ^
  -DBLA_STATIC=ON ^
  -DMKL_LINK=static ^
  -DMKL_INTERFACE_FULL=intel_lp64 ^
  -DMUMPS_DIR="%MUMPS_DIR%" ^
  -DMETIS5_DIR="%METIS5_DIR%" ^
  -UOPENSEES_METIS5_LIBRARY ^
  -UOPENMPI ^
  -DPARALLEL_PROCESSING=OFF ^
  -DCMAKE_NINJA_FORCE_RESPONSE_FILE=ON ^
  -DCMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS=ON ^
  -DCMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS=ON ^
  -DCMAKE_Fortran_USE_RESPONSE_FILE_FOR_OBJECTS=ON

"%CMAKE%" --build build-mp\Release --target OpenSeesMP --parallel %JOBS%
```

#### 4. Stage Tcl runtime for OpenSeesMP (required for `mpiexec`)

OpenSeesMP loads **`tcl86t.dll`** from ActiveTcl (`%TCL_ROOT%\bin\`, copied next to the exe at
link time). The **`init.tcl`** script tree must be the **same Tcl patch** as that DLL.

Do **not** copy only Conan’s `tcl/8.6.11` scripts while the exe ships ActiveTcl **8.6.16** — you
will get `version conflict for package "Tcl": have 8.6.16, need exactly 8.6.11`.

Copy **ActiveTcl** scripts and DLL from `%TCL_ROOT%` (same patch as `tcl86t.dll`):

```bat
cd /d "%PARENT%\OpenSees"

if not exist build-mp\lib mkdir build-mp\lib
robocopy "%TCL_ROOT%\lib\tcl8.6" "build-mp\lib\tcl8.6" /E
if exist "%TCL_ROOT%\lib\tcl8" robocopy "%TCL_ROOT%\lib\tcl8" "build-mp\lib\tcl8" /E

if not exist lib mkdir lib
robocopy "%TCL_ROOT%\lib\tcl8.6" "lib\tcl8.6" /E

copy /Y "%TCL_ROOT%\bin\tcl86t.dll" "build-mp\Release\"
```

This populates:

- `build-mp\lib\tcl8.6\` — primary runtime path for OpenSeesMP
- `lib\tcl8.6\` — also searched when running from the repo tree
- `build-mp\Release\tcl86t.dll` — refreshed to match `init.tcl`

Re-run these copies after every OpenSeesMP rebuild or ActiveTcl upgrade.

**Verify** (the `package require` line must match your `tcl86t.dll`):

```bat
findstr "package require -exact Tcl" "%PARENT%\OpenSees\build-mp\lib\tcl8.6\init.tcl"
dir "%PARENT%\OpenSees\build-mp\Release\OpenSeesMP.exe"
dir "%PARENT%\OpenSees\build-mp\Release\tcl86t.dll"
dir "%PARENT%\OpenSees\build-mp\lib\tcl8.6\init.tcl"
```

---

### OpenSees and OpenSeesMP (with CUDA)

Install CUDA 12.x first. Add `nvcc` to `PATH` and pass `-DCUDAToolkit_ROOT=...` on **both**
configure lines below.

```bat
set CUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
set PATH=%CUDAToolkit_ROOT%\bin;%PATH%

set CUDA_CMAKE_ARGS=-DCUDAToolkit_ROOT="%CUDAToolkit_ROOT%" -Ucudss_DIR -Ucudss_INCLUDE_DIR -Ucudss_LIBRARY_DIR -Ucudss_BINARY_DIR -UAMGX_NO_MPI_DIR
```

Run the **Conan** step from the CPU-only section (once per tree), then configure with the
extra CUDA flags appended:

```bat
"%CMAKE%" -S . -B build\Release -G Ninja ^
  -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN_OP%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
  -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DCMAKE_Fortran_COMPILER=ifx ^
  -DBLA_STATIC=ON ^
  -DMKL_LINK=static ^
  -DMKL_INTERFACE_FULL=intel_lp64 ^
  -DMUMPS_DIR="%MUMPS_DIR%" ^
  -DMETIS5_DIR="%METIS5_DIR%" ^
  -UOPENSEES_METIS5_LIBRARY ^
  -UOPENMPI ^
  -DPARALLEL_PROCESSING=OFF ^
  %CUDA_CMAKE_ARGS% ^
  -DCMAKE_NINJA_FORCE_RESPONSE_FILE=ON ^
  -DCMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS=ON ^
  -DCMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS=ON ^
  -DCMAKE_Fortran_USE_RESPONSE_FILE_FOR_OBJECTS=ON

"%CMAKE%" --build build\Release --target OpenSees --parallel %JOBS%

"%CMAKE%" -S . -B build-mp\Release -G Ninja ^
  -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN_MP%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
  -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" ^
  -DCMAKE_Fortran_COMPILER=ifx ^
  -DBLA_STATIC=ON ^
  -DMKL_LINK=static ^
  -DMKL_INTERFACE_FULL=intel_lp64 ^
  -DMUMPS_DIR="%MUMPS_DIR%" ^
  -DMETIS5_DIR="%METIS5_DIR%" ^
  -UOPENSEES_METIS5_LIBRARY ^
  -UOPENMPI ^
  -DPARALLEL_PROCESSING=OFF ^
  %CUDA_CMAKE_ARGS% ^
  -DCMAKE_NINJA_FORCE_RESPONSE_FILE=ON ^
  -DCMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS=ON ^
  -DCMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS=ON ^
  -DCMAKE_Fortran_USE_RESPONSE_FILE_FOR_OBJECTS=ON

"%CMAKE%" --build build-mp\Release --target OpenSeesMP --parallel %JOBS%
```

Stage Tcl for OpenSeesMP as in [step 4 of the CPU-only section](#4-stage-tcl-runtime-for-openseesmp-required-for-mpiexec).

**Verify:**

```bat
dir "%PARENT%\OpenSees\build\Release\OpenSees.exe"
dir "%PARENT%\OpenSees\build-mp\Release\OpenSeesMP.exe"
dir "%PARENT%\OpenSees\build-mp\lib\tcl8.6\init.tcl"
nvcc --version
```

---

### Stage OpenFresco libraries

OpenFresco **Release-CMake** links against static `.lib` files flattened into
`build\lib`. Run this after **OpenSees.exe** has been built.

```bat
cd /d "%PARENT%\OpenSees"
set REL=build\Release
set STAGE=build\lib
set BINS=build\bin-fresco

if not exist "%STAGE%" mkdir "%STAGE%"
if not exist "%BINS%" mkdir "%BINS%"

"%CMAKE%" --build "%REL%" --target ITPACK --parallel %JOBS%

copy /Y "%REL%\OTHER\ARPACK\ARPACK.lib" "%STAGE%\"
copy /Y "%REL%\OTHER\SuperLU_5.1.1\SUPERLU.lib" "%STAGE%\"
copy /Y "%REL%\OTHER\UMFPACK\UMFPACK.lib" "%STAGE%\"
copy /Y "%REL%\OTHER\AMD\AMD.lib" "%STAGE%\"
copy /Y "%REL%\OTHER\CSPARSE\CSPARSE.lib" "%STAGE%\"
copy /Y "%REL%\OTHER\ITPACK\ITPACK.lib" "%STAGE%\"
copy /Y "%REL%\OTHER\tetgen1.4.3\tet.lib" "%STAGE%\"
copy /Y "%REL%\OTHER\Triangle\triangle.lib" "%STAGE%\"
copy /Y "%REL%\SRC\material\uniaxial\OPS_Material_f.lib" "%STAGE%\"
copy /Y "%REL%\SRC\material\nD\feap\OPS_Material_nD_Feap_f.lib" "%STAGE%\"
copy /Y "%REL%\SRC\material\uniaxial\drain\OPS_Material_Uniaxial_Drain_f.lib" "%STAGE%\"
copy /Y "%REL%\SRC\system_of_eqn\linearSOE\sparseSYM\OPS_SysOfEqn_f.lib" "%STAGE%\"

REM Optional: copy a runtime bundle (same names as the build outputs)
copy /Y "%REL%\OpenSees.exe" "%BINS%\OpenSees.exe"
copy /Y "%TCL_ROOT%\bin\tcl86t.dll" "%BINS%\"
copy /Y "build-mp\Release\OpenSeesMP.exe" "%BINS%\OpenSeesMP.exe"
```

**Verify:**

```bat
dir "%PARENT%\OpenSees\build\lib\ARPACK.lib"
dir "%PARENT%\OpenSees\build\Release\OpenSees.exe"
dir "%PARENT%\OpenSees\build-mp\Release\OpenSeesMP.exe"
```

Primary build outputs (CPU or CUDA):

```text
%PARENT%\OpenSees\build\Release\OpenSees.exe      <- serial / loadPackage
%PARENT%\OpenSees\build-mp\Release\OpenSeesMP.exe <- MPI / partition / loadPackage
%PARENT%\OpenSees\build\lib\*.lib                 <- OpenFresco Release-CMake link
%PARENT%\OpenSees\build\bin-fresco\               <- optional copied runtime bundle
```

On the **RTHS-CUDA** branch these executables already use dynamic **`tcl86t.dll`** and work
with OpenFresco via `loadPackage`. There is no separate `OpenSeesFresco.exe` rename step.

---

### OpenFresco plugin

**Important:** use configuration **`Release-CMake`**, not plain **`Release`**.

OpenFresco links against OpenSees static libraries from **`build\lib`** via
`WIN64\proj\OpenSees-CMake.props` (default `OpenSeesRoot` → sibling `%OPENSEES_DIR%`).

```bat
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 mod

cd /d "%PARENT%\%OPENFRESCO_DIR%\WIN64"

"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" ^
  OpenFresco.sln ^
  /p:Configuration=Release-CMake ^
  /p:Platform=x64 ^
  /t:OpenFrescoTcl_dll ^
  /m
```

Expected output:

```text
%PARENT%\%OPENFRESCO_DIR%\WIN64\bin-cmake\OpenFrescoTcl.dll
```

If OpenSees is **not** at `%PARENT%\OpenSees`, override the root:

```bat
MSBuild OpenFresco.sln ^
  /p:Configuration=Release-CMake ^
  /p:Platform=x64 ^
  /t:OpenFrescoTcl_dll ^
  /p:OpenSeesRoot=C:\full\path\to\OpenSees ^
  /m
```

OpenSSL must be at `%OPENSSL_ROOT%` (OpenFresco vcxproj files use
`include` and `lib\VC\x64\MT` for **Release-CMake**).

**Verify:**

```bat
dir "%PARENT%\OpenFresco\WIN64\bin-cmake\OpenFrescoTcl.dll"
```

---

## Verify installation

### loadPackage smoke test

```bat
set EX=%PARENT%\%OPENSEES_DIR%\EXAMPLES\ShearBuilding40SP
set BIN=%PARENT%\%OPENSEES_DIR%\build\Release
set OF=%PARENT%\%OPENFRESCO_DIR%\WIN64\bin-cmake

copy /Y "%BIN%\OpenSees.exe" "%EX%\"
copy /Y "C:\Program Files\Tcl\bin\tcl86t.dll" "%EX%\"
copy /Y "%OF%\OpenFrescoTcl.dll" "%EX%\"
copy /Y "C:\Program Files\OpenSSL\bin\libssl-3-x64.dll" "%EX%\" 2>nul
copy /Y "C:\Program Files\OpenSSL\bin\libcrypto-3-x64.dll" "%EX%\" 2>nul

cd /d "%EX%"
set TCL_LIBRARY=C:\Program Files\Tcl\lib\tcl8.6

OpenSees.exe -version
```

Create a tiny probe script `%EX%\loadPackage_probe.tcl`:

```tcl
wipe
model BasicBuilder -ndm 1 -ndf 1
loadPackage OpenFrescoTcl
puts "loadPackage OpenFrescoTcl OK"
exit 0
```

Run:

```bat
set TCL_LIBRARY=C:\Program Files\Tcl\lib\tcl8.6
OpenSees.exe loadPackage_probe.tcl
```

You should see OpenFresco version text and `loadPackage OpenFrescoTcl OK`.

### End-to-end checklist

| Step | Verify command |
|------|----------------|
| MUMPS | `dir %PARENT%\mumps\build\dmumps.lib` |
| METIS 5 | `dir %PARENT%\metis-5.1.0\install\lib\metis.lib` |
| OpenSees | `dir %PARENT%\OpenSees\build\Release\OpenSees.exe` |
| OpenSeesMP | `dir %PARENT%\OpenSees\build-mp\Release\OpenSeesMP.exe` |
| OpenFresco staging | `dir %PARENT%\OpenSees\build\lib\ARPACK.lib` |
| OpenFresco plugin | `loadPackage OpenFrescoTcl` probe (above) |

### ShearBuilding40 example (optional)

```bat
cd /d "%PARENT%\OpenSees\EXAMPLES\ShearBuilding40SP"
set SHEAR40_MODE=local
set TCL_LIBRARY=%TCL_ROOT%\lib\tcl8.6

OpenSees.exe ShearBuilding40.tcl
```

Copy `OpenFrescoTcl.dll`, `tcl86t.dll`, and OpenSSL DLLs beside `OpenSees.exe` first if they
are not already in that folder (see [Built executables and usage](#built-executables-and-usage)).

---

## Built executables and usage

After a full build, these are the files you run.

### Where executables land

| Program | Built path | Role |
|---------|------------|------|
| **`OpenSees.exe`** | `%PARENT%\OpenSees\build\Release\OpenSees.exe` | Serial Tcl interpreter; hybrid simulation with `loadPackage OpenFrescoTcl` |
| **`OpenSeesMP.exe`** | `%PARENT%\OpenSees\build-mp\Release\OpenSeesMP.exe` | MPI parallel Tcl; needs `mpiexec`, `build-mp\lib\tcl8.6`, and matching `tcl86t.dll` |
| **`OpenFrescoTcl.dll`** | `%PARENT%\OpenFresco\WIN64\bin-cmake\OpenFrescoTcl.dll` | OpenFresco plugin loaded at run time |

Both OpenSees executables on **RTHS-CUDA** link **`tcl86t.dll`** dynamically (not a separate
“Fresco” binary). Use the names **`OpenSees.exe`** and **`OpenSeesMP.exe`** only.

Optional convenience copies (if you ran [Stage OpenFresco libraries](#building--stage-openfresco-libraries)):

```text
%PARENT%\OpenSees\build\bin-fresco\OpenSees.exe
%PARENT%\OpenSees\build\bin-fresco\OpenSeesMP.exe
%PARENT%\OpenSees\build\bin-fresco\tcl86t.dll
```

### Runtime folder for hybrid simulation

For any example or test folder, put these files **in the same directory** (or on `PATH`):

| File | Typical source |
|------|----------------|
| `OpenSees.exe` (or `OpenSeesMP.exe`) | `build\Release\` or `build-mp\Release\` |
| `OpenFrescoTcl.dll` | `OpenFresco\WIN64\bin-cmake\` |
| `tcl86t.dll` | `%TCL_ROOT%\bin\` (serial) or `build-mp\Release\` (MPI) |
| `lib\tcl8.6\` (OpenSeesMP only) | `build-mp\lib\tcl8.6\` — set `TCL_LIBRARY` to this folder |
| `libssl-3-x64.dll`, `libcrypto-3-x64.dll` | `%OPENSSL_ROOT%\bin\` (if needed at run time) |

Set Tcl scripts for the interpreter:

```bat
REM Serial OpenSees / OpenFresco:
set TCL_LIBRARY=%TCL_ROOT%\lib\tcl8.6

REM OpenSeesMP (after staging — see OpenSeesMP — Tcl runtime):
set TCL_LIBRARY=%PARENT%\OpenSees\build-mp\lib\tcl8.6
```

### How to run

**Check versions:**

```bat
cd /d "%PARENT%\OpenSees\build\Release"
set TCL_LIBRARY=%TCL_ROOT%\lib\tcl8.6
OpenSees.exe -version
```

**loadPackage smoke test** (from [Verify installation](#verify-installation)) uses the same
`OpenSees.exe`.

**Serial hybrid example** (ShearBuilding40, local mode):

```bat
cd /d "%PARENT%\OpenSees\EXAMPLES\ShearBuilding40SP"
copy /Y "%PARENT%\OpenSees\build\Release\OpenSees.exe" .
copy /Y "%PARENT%\OpenFresco\WIN64\bin-cmake\OpenFrescoTcl.dll" .
copy /Y "%TCL_ROOT%\bin\tcl86t.dll" .
copy /Y "%OPENSSL_ROOT%\bin\libssl-3-x64.dll" . 2>nul
copy /Y "%OPENSSL_ROOT%\bin\libcrypto-3-x64.dll" . 2>nul

set TCL_LIBRARY=%TCL_ROOT%\lib\tcl8.6
set SHEAR40_MODE=local
OpenSees.exe ShearBuilding40.tcl
```

**MPI example** (OpenSeesMP):

```bat
cd /d "%PARENT%\OpenSees\EXAMPLES\ShearBuilding40SP"
copy /Y "%PARENT%\OpenSees\build-mp\Release\OpenSeesMP.exe" .
copy /Y "%PARENT%\OpenSees\build-mp\Release\tcl86t.dll" .
REM copy OpenFrescoTcl.dll, OpenSSL DLLs if needed

set TCL_LIBRARY=%PARENT%\OpenSees\build-mp\lib\tcl8.6
call "%ONEAPI_SETVARS%" intel64 mod
set I_MPI_PIN=on
set I_MPI_PIN_CELL=core
set I_MPI_FABRICS=shm
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
mpiexec -n 4 OpenSeesMP.exe ShearBuilding40MP.tcl
```

Use Intel MPI’s `mpiexec` from oneAPI. For single-node MUMPS jobs, also set the
[Intel MPI runtime](#intel-mpi-runtime-windows-single-node) knobs (`I_MPI_PIN=on`,
`I_MPI_FABRICS=shm`, `OMP_NUM_THREADS=1`, …). Partitioning examples live under
`EXAMPLES\Partition` and `EXAMPLES\LargeMP`.

**OpenFresco local examples** (e.g. `OpenFresco\EXAMPLES\OneBayFrame\OpenSees\OneBayFrame_Local.tcl`):

```bat
cd /d "%PARENT%\OpenFresco\EXAMPLES\OneBayFrame\OpenSees"
copy /Y "%PARENT%\OpenSees\build\Release\OpenSees.exe" .
copy /Y "%PARENT%\OpenFresco\WIN64\bin-cmake\OpenFrescoTcl.dll" .
copy /Y "%TCL_ROOT%\bin\tcl86t.dll" .

set TCL_LIBRARY=%TCL_ROOT%\lib\tcl8.6
OpenSees.exe OneBayFrame_Local.tcl
```

Run from the example directory so relative data files (e.g. `elcentro.txt`) resolve.

---

## Intel MPI runtime (Windows, single node)

OpenSeesMP + MUMPS on a **small explicit** model (frequent tiny MPI messages) is often much
slower on native Windows than the same job in WSL. That is Intel MPI latency, not a wrong
solver. These environment variables helped a lot on oneAPI (IPL2) with 4 ranks on one PC.

Set them in the **same cmd window** as `mpiexec` (after `setvars.bat`):

```bat
set I_MPI_PIN=on
set I_MPI_PIN_CELL=core
set I_MPI_FABRICS=shm
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
```

| Variable | Why |
|----------|-----|
| `I_MPI_PIN=on` | Keep each rank on a fixed CPU |
| `I_MPI_PIN_CELL=core` | One rank per **physical core** (not two hyperthreads on the same core) |
| `I_MPI_FABRICS=shm` | Shared-memory only — skip OFI/TCP for intra-node messages |
| `OMP_NUM_THREADS=1` | OpenSeesMP links threaded MKL + MUMPS `/Qiopenmp`; extra OpenMP threads steal cores from other ranks |
| `MKL_NUM_THREADS=1` | Same, for MKL |

Do **not** use `set I_MPI_PIN=1` or `I_MPI_PIN_DOMAIN=core` on this oneAPI. Newer Intel MPI
treats `I_MPI_PIN=1` as an affinity list and prints `IPL2 Error: ... error parsing affinity list`
once per rank, then runs **without** pinning.

You can pass the same knobs on the `mpiexec` line:

```bat
mpiexec -genv I_MPI_PIN on -genv I_MPI_PIN_CELL core -genv I_MPI_FABRICS shm ^
  -genv OMP_NUM_THREADS 1 -genv MKL_NUM_THREADS 1 ^
  -n 4 "%PARENT%\OpenSees\build-mp\Release\OpenSeesMP.exe" RunParallel.tcl
```

**Optional:** `I_MPI_SPIN_COUNT=20000` makes ranks busy-wait longer for the next MPI message
(lower latency, hotter idle CPU). Default is already ~2000 when you are not oversubscribed.
Try it only after the table above; it will not match the shm/pin/threads=1 gains. Do **not**
set `I_MPI_WAIT_MODE=1` (that is the oversubscription/yield path and is slower here).

Confirm shm with `set I_MPI_DEBUG=2` once (startup lines should mention `shm`), then unset it.

---

## OpenSeesMP — Tcl runtime after build

OpenSeesMP needs **`tcl86t.dll`** and **`lib\tcl8.6\`** at run time. Both must come from the
**same ActiveTcl install** (`%TCL_ROOT%`, default `C:\Program Files\Tcl`).

| Piece | Location | Source |
|-------|----------|--------|
| **`tcl86t.dll`** | `build-mp\Release\` (beside `OpenSeesMP.exe`) | `%TCL_ROOT%\bin\` (post-build copy) |
| **Tcl scripts** | `build-mp\lib\tcl8.6\` and repo `lib\tcl8.6\` | `%TCL_ROOT%\lib\tcl8.6\` — see [step 4](#4-stage-tcl-runtime-for-openseesmp-required-for-mpiexec) |

**Stage after every OpenSeesMP build** (or when Tcl errors appear):

```bat
cd /d "%PARENT%\OpenSees"

if not exist build-mp\lib mkdir build-mp\lib
robocopy "%TCL_ROOT%\lib\tcl8.6" "build-mp\lib\tcl8.6" /E
if exist "%TCL_ROOT%\lib\tcl8" robocopy "%TCL_ROOT%\lib\tcl8" "build-mp\lib\tcl8" /E
if not exist lib mkdir lib
robocopy "%TCL_ROOT%\lib\tcl8.6" "lib\tcl8.6" /E
copy /Y "%TCL_ROOT%\bin\tcl86t.dll" "build-mp\Release\"
```

**Run** from the **model directory**. Point `mpiexec` at `build-mp\Release\OpenSeesMP.exe` (do **not** copy the exe into the model folder). `tcl86t.dll` loads from that same `Release` folder. Set `TCL_LIBRARY` to the staged scripts:

```bat
set TCL_LIBRARY=%PARENT%\OpenSees\build-mp\lib\tcl8.6
```

**MPI runs:** use a **local Windows path** (`cd /d C:\...`), not `\\wsl.localhost\...`.

**Example — OSU SSI Bridge:**

```bat
call "%ONEAPI_SETVARS%" intel64 mod
cd /d C:\Users\YOURNAME\OpenSees_Runs\OSU_SSI_Bridge
set TCL_LIBRARY=%PARENT%\OpenSees\build-mp\lib\tcl8.6
set I_MPI_PIN=on
set I_MPI_PIN_CELL=core
set I_MPI_FABRICS=shm
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
mpiexec -n 6 "%PARENT%\OpenSees\build-mp\Release\OpenSeesMP.exe" RunParallel.tcl
```

See [Intel MPI runtime](#intel-mpi-runtime-windows-single-node) for what these do.

---

## Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| `cannot open input file 'dmumps.lib'` (OpenSees) | Build MUMPS first — see [Building — MUMPS](#building--mumps); keep full `mumps\build` tree with `_deps\` |
| `dmumps_c.h` not found (OpenSees) | Point `MUMPS_DIR` at `mumps\build`, not a folder containing only `.lib` files |
| `cannot open input file 'cudadevrt.lib'` | Pull latest `RTHS-CUDA` OpenSees; rebuild after CUDA `/LIBPATH` fix |
| `cannot open input file 'ARPACK.lib'` (OpenFresco) | Complete [Stage OpenFresco libraries](#building--stage-openfresco-libraries); build **Release-CMake**, not Release |
| `openssl/ssl.h` not found | Install OpenSSL at `%OPENSSL_ROOT%`; OpenFresco branch must have `OpenSSL\include` paths |
| `Can't find init.tcl` | Set `TCL_LIBRARY` to a folder containing `init.tcl`. OpenSeesMP: use `build-mp\lib\tcl8.6` after [step 4](#4-stage-tcl-runtime-for-openseesmp-required-for-mpiexec) |
| `package require -exact Tcl` version error | Re-run [step 4](#4-stage-tcl-runtime-for-openseesmp-required-for-mpiexec): copy `%TCL_ROOT%\lib\tcl8.6` to `build-mp\lib`, not Conan 8.6.11 scripts alone |
| `The command line is too long` (Fortran / ifx) | Add `-DCMAKE_Fortran_USE_RESPONSE_FILE_FOR_OBJECTS=ON` to CMake configure (with the other `RESPONSE_FILE` flags) |
| `partition` stub in OpenSeesMP | Build METIS 5 — see [Building — METIS 5](#building--metis-5) |
| `mpiexec` / `OpenSeesMP.exe` not found on UNC path | Run from `cd /d C:\...` (local path), not `\\wsl.localhost\...` |
| Stale CMake cache (`/home/...` in `CMakeCache.txt`) | Delete `build-mp\Release` (or `build\Release`) and reconfigure on Windows |
| Conan not found | Create conda env **`RTHS-CUDA`**, `pip install conan`, add `%CONDA_ENV%\Scripts` to PATH |
| CUDA configure fails / `cudadevrt.lib` missing | Set `CUDAToolkit_ROOT=%CUDA_ROOT%`, add `%CUDA_ROOT%\bin` to PATH, re-run cmake configure on both trees |
| Wrong OpenSees folder name | Clone OpenSees into **`OpenSees`** under `%PARENT%`, or pass `/p:OpenSeesRoot=...` to MSBuild |
| `IPL2 Error: ... error parsing affinity list` | Do not set `I_MPI_PIN=1`. Use `I_MPI_PIN=on` and `I_MPI_PIN_CELL=core`, or clear `I_MPI_PIN` / `I_MPI_PIN_DOMAIN` — see [Intel MPI runtime](#intel-mpi-runtime-windows-single-node) |
| OpenSeesMP + MUMPS much slower than WSL (small explicit model) | Set `I_MPI_PIN=on`, `I_MPI_PIN_CELL=core`, `I_MPI_FABRICS=shm`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. Pinning alone will not match WSL on a tiny mesh |

---

## Appendix: full copy-paste script

Adjust `PARENT` and `CONDA_ROOT` once. Run **either** the CPU-only OpenSees block **or**
the CUDA block — not both on the same build tree unless you reconfigure cleanly.

### Shared setup (MUMPS + METIS + clones)

```bat
set PARENT=C:\Users\YOURNAME\source\repos\simpsoba\RTHS-CUDA
set CONDA_ROOT=C:\Users\YOURNAME\AppData\Local\anaconda3
set CONDA_ENV=%CONDA_ROOT%\envs\RTHS-CUDA

call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 mod
set PATH=%CONDA_ENV%;%CONDA_ENV%\Scripts;%PATH%
set "CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9

mkdir "%PARENT%"
cd /d "%PARENT%"

git clone -b RTHS-CUDA https://github.com/simpsoba/OpenSees.git OpenSees
git clone -b RTHS-CUDA https://github.com/simpsoba/OpenFresco.git OpenFresco

git clone https://github.com/OpenSees/mumps.git mumps
cd mumps
if not exist build mkdir build
cd build
cmake .. -G Ninja -Darith=d -Dopenmp=OFF -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DBUILD_TESTING=OFF -DCMAKE_Fortran_COMPILER=ifx -DCMAKE_C_COMPILER=icx -DCMAKE_C_FLAGS="/DWIN32 /D_WINDOWS /Qiopenmp" -DCMAKE_Fortran_FLAGS="/nologo /fpp /Qiopenmp" -DCMAKE_EXE_LINKER_FLAGS="/Qiopenmp" -DCMAKE_C_FLAGS_RELEASE="/O3 /Ob2 /DNDEBUG" -DCMAKE_Fortran_FLAGS_RELEASE="/O3 /DNDEBUG" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --parallel 8
cd ..\..

git clone --depth 1 https://github.com/eric2003/METIS-5.1.0-Modified.git metis-5.1.0
cd metis-5.1.0
set CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set PREFIX=%CD%\install
"%CMAKE%" -S . -B build -G "Visual Studio 17 2022" -A x64 -DMETIS_ENABLE_64BIT=OFF -DSHARED=FALSE -DCMAKE_INSTALL_PREFIX="%PREFIX%" -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
"%CMAKE%" --build build --config Release --target metis --parallel 8
mkdir install\include install\lib
copy include\metis.h install\include\
copy build\libmetis\Release\metis.lib install\lib\
cd ..
```

### OpenSees and OpenSeesMP — **without CUDA**

```bat
cd /d "%PARENT%\OpenSees"

conan install . -of build -s build_type=Release -s arch=x86_64 -s compiler.runtime=static --build=missing -c tools.cmake.cmaketoolchain:generator=Ninja
conan install . -of build-mp -s build_type=Release -s arch=x86_64 -s compiler.runtime=static --build=missing -c tools.cmake.cmaketoolchain:generator=Ninja

set MUMPS_DIR=%PARENT%\mumps\build
set METIS5_DIR=%PARENT%\metis-5.1.0\install
set TCL_ROOT=C:\Program Files\Tcl
set TOOLCHAIN_OP=%CD%\build\Release\generators\conan_toolchain.cmake
if not exist "%TOOLCHAIN_OP%" set TOOLCHAIN_OP=%CD%\build\build\Release\generators\conan_toolchain.cmake
set TOOLCHAIN_MP=%CD%\build-mp\Release\generators\conan_toolchain.cmake
if not exist "%TOOLCHAIN_MP%" set TOOLCHAIN_MP=%CD%\build-mp\build\Release\generators\conan_toolchain.cmake

"%CMAKE%" -S . -B build\Release -G Ninja -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN_OP%" -DCMAKE_BUILD_TYPE=Release -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" -DCMAKE_Fortran_COMPILER=ifx -DBLA_STATIC=ON -DMKL_LINK=static -DMKL_INTERFACE_FULL=intel_lp64 -DMUMPS_DIR="%MUMPS_DIR%" -DMETIS5_DIR="%METIS5_DIR%" -UOPENSEES_METIS5_LIBRARY -UOPENMPI -DPARALLEL_PROCESSING=OFF -DCMAKE_NINJA_FORCE_RESPONSE_FILE=ON -DCMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS=ON -DCMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS=ON -DCMAKE_Fortran_USE_RESPONSE_FILE_FOR_OBJECTS=ON
"%CMAKE%" --build build\Release --target OpenSees --parallel 10

"%CMAKE%" -S . -B build-mp\Release -G Ninja -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN_MP%" -DCMAKE_BUILD_TYPE=Release -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DCMAKE_EXE_LINKER_FLAGS="/FORCE:MULTIPLE" -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" -DCMAKE_Fortran_COMPILER=ifx -DBLA_STATIC=ON -DMKL_LINK=static -DMKL_INTERFACE_FULL=intel_lp64 -DMUMPS_DIR="%MUMPS_DIR%" -DMETIS5_DIR="%METIS5_DIR%" -UOPENSEES_METIS5_LIBRARY -UOPENMPI -DPARALLEL_PROCESSING=OFF -DCMAKE_NINJA_FORCE_RESPONSE_FILE=ON -DCMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS=ON -DCMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS=ON -DCMAKE_Fortran_USE_RESPONSE_FILE_FOR_OBJECTS=ON
"%CMAKE%" --build build-mp\Release --target OpenSeesMP --parallel 10

if not exist build-mp\lib mkdir build-mp\lib
robocopy "%TCL_ROOT%\lib\tcl8.6" "build-mp\lib\tcl8.6" /E
if exist "%TCL_ROOT%\lib\tcl8" robocopy "%TCL_ROOT%\lib\tcl8" "build-mp\lib\tcl8" /E
if not exist lib mkdir lib
robocopy "%TCL_ROOT%\lib\tcl8.6" "lib\tcl8.6" /E
copy /Y "%TCL_ROOT%\bin\tcl86t.dll" "build-mp\Release\"
```

### OpenSees and OpenSeesMP — **with CUDA**

Append `-DCUDAToolkit_ROOT="%CUDA_ROOT%" -Ucudss_DIR -Ucudss_INCLUDE_DIR -Ucudss_LIBRARY_DIR -Ucudss_BINARY_DIR -UAMGX_NO_MPI_DIR` to both `"%CMAKE%" -S . -B ...` configure lines above, and set `PATH=%CUDA_ROOT%\bin;%PATH%` before configuring.

### Stage OpenFresco libraries

See [Building — stage OpenFresco libraries](#building--stage-openfresco-libraries).

### OpenFresco plugin (same for CPU and CUDA OpenSees builds)

```bat
cd /d "%PARENT%\OpenFresco\WIN64"
"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" OpenFresco.sln /p:Configuration=Release-CMake /p:Platform=x64 /t:OpenFrescoTcl_dll /m
```

---

## Related docs

- [EXAMPLES/ShearBuilding40SP/README.md](EXAMPLES/ShearBuilding40SP/README.md) — hybrid example layout
