"""
Import helper for OpenSees Python bindings.

Preference order:
1) The locally built extension at <repo>/build/Release/opensees.so
   (imported as: `import opensees as ops`)
2) Fallback to OpenSeesPy (imported as: `import openseespy.opensees as ops`)

This keeps the EXAMPLES runnable while ensuring you test the code you just built.

Import order (local build only):
  Import this module before numpy, scipy, or pandas. The locally built opensees.so
  is linked with -static-libstdc++; if those packages initialize first, PathSeries
  reads from -filePath silently fail and UniformExcitation cannot attach the accel
  series. Pip OpenSeesPy is unaffected (shared libstdc++).
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))


def _runtime_dll_dirs(repo_root: str) -> list[str]:
    cuda_root = os.environ.get(
        "CUDAToolkit_ROOT",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9",
    )
    cudss_root = os.environ.get(
        "OPENSEES_CUDSS_DIR",
        r"C:\Program Files\NVIDIA cuDSS\v0.8",
    )
    oneapi_bin = os.environ.get(
        "ONEAPI_COMPILER_BIN",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin",
    )
    return [
        os.path.join(repo_root, "build", "Release"),
        os.path.join(cudss_root, "bin", "12"),
        os.path.join(cuda_root, "bin"),
        oneapi_bin,
    ]


def _add_local_build_to_sys_path() -> None:
    repo_root = _repo_root()
    build_release = os.path.join(repo_root, "build", "Release")
    if os.path.isdir(build_release) and build_release not in sys.path:
        sys.path.insert(0, build_release)
    if sys.platform == "win32":
        runtime_dirs = _runtime_dll_dirs(repo_root)
        prefix = os.pathsep.join(p for p in runtime_dirs if os.path.isdir(p))
        if prefix:
            os.environ["PATH"] = prefix + os.pathsep + os.environ.get("PATH", "")
        for path in runtime_dirs:
            if os.path.isdir(path):
                os.add_dll_directory(path)


_add_local_build_to_sys_path()

ops: Any

try:
    # locally built module (build/Release/opensees.so)
    import opensees as ops  # type: ignore
    _ops_src = "local build (opensees)"
except Exception:
    # fallback to OpenSeesPy
    import openseespy.opensees as ops  # type: ignore
    _ops_src = "OpenSeesPy (openseespy.opensees)"
else:
    for _early_mod in ("numpy", "scipy", "pandas"):
        if _early_mod in sys.modules:
            warnings.warn(
                f"{_early_mod} was imported before OpenSees; PathSeries -filePath may fail "
                "with the locally built opensees.so. Import ops_import (or opensees) first.",
                stacklevel=2,
            )
            break

try:
    _ops_file = getattr(ops, "__file__", None)
except Exception:
    _ops_file = None

print(f"[ops_import] using {_ops_src}: {_ops_file}")

