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


def _add_local_build_to_sys_path() -> None:
    # EXAMPLES/KRAlphaExplicit/ops_import.py -> repo root is ../..
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    build_release = os.path.join(repo_root, "build", "Release")
    if os.path.isdir(build_release) and build_release not in sys.path:
        sys.path.insert(0, build_release)


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

