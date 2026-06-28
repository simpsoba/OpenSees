"""Run cuda_explicit_alpha_tp_smoke.py after setvars; cuDSS/CUDA on PATH + add_dll_directory for Python."""
import os
import runpy
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
release = os.path.join(root, "build", "Release")
cuda_root = os.environ.get("CUDAToolkit_ROOT", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9")
cudss_root = os.environ.get("CUDSS_DIR", r"C:\Program Files\NVIDIA cuDSS\v0.8")
oneapi_bin = os.environ.get(
    "ONEAPI_COMPILER_BIN",
    r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin",
)
runtime_dirs = [
    release,
    os.path.join(cudss_root, "bin", "12"),
    os.path.join(cuda_root, "bin"),
    oneapi_bin,
]
prefix = os.pathsep.join(p for p in runtime_dirs if os.path.isdir(p))
if prefix:
    os.environ["PATH"] = prefix + os.pathsep + os.environ.get("PATH", "")
if sys.platform == "win32":
    for path in runtime_dirs:
        if os.path.isdir(path):
            os.add_dll_directory(path)

sys.path.insert(0, release)
runpy.run_path(os.path.join(root, "tests", "cuda_explicit_alpha_tp_smoke.py"), run_name="__main__")
