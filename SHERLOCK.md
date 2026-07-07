# OpenSees on Stanford Sherlock

Quick reference for running a build produced by `makeSherlock.sh` in this repo.

## Build (once)

From the repo root on a dev/compute node (`sh_dev`), not the login node:

```bash
chmod +x makeSherlock.sh
./makeSherlock.sh
```

Outputs:

| File | Description |
|------|-------------|
| `build/Release/OpenSees` | Sequential Tcl interpreter |
| `build/Release/opensees.so` | Python module (copy of `OpenSeesPy.so`) |

## Run OpenSees (Tcl)

Load the GCC module so `libgfortran` is available (required at runtime):

```bash
cd $GROUP_HOME/opensees-fatigue
module load gcc/12.4.0

./build/Release/OpenSees yourModel.tcl
```

## Run OpenSeesPy (Python)

Use the same Python version the build used:

```bash
module load gcc/12.4.0 python/3.12.1

python3 -c "
import sys
sys.path.insert(0, '/home/groups/bsimpson/opensees-fatigue/build/Release')
import opensees as ops
print('OpenSeesPy OK')
"
```

Or in a script:

```python
import sys
sys.path.insert(0, "/home/groups/bsimpson/opensees-fatigue/build/Release")
import opensees as ops
```

## Submit a job (example)

```bash
#!/bin/bash
#SBATCH --job-name=opensees
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

module load gcc/12.4.0 python/3.12.1
cd $GROUP_HOME/opensees-fatigue
./build/Release/OpenSees examples/yourModel.tcl
python3 -c "
import sys
sys.path.insert(0, '/home/groups/bsimpson/opensees-fatigue/build/Release')
import opensees as ops
print('OpenSeesPy OK')
"
```

## Troubleshooting

**`libgfortran.so.5: cannot open shared object file`**

```bash
module load gcc/12.4.0
```
