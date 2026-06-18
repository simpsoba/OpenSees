# -*- coding: utf-8 -*-
"""Shared integrator-matrix runner for 1-D shear-frame examples (ops-cuda layout)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

RunSpec = Tuple[str, str, List[Union[float, str]], Optional[str]]

DEFAULT_RHOS: List[float] = [1.0, 0.5]


def _repo_root(here: Path) -> Path:
    return (here / ".." / ".." / "..").resolve()


def _python_env(here: Path) -> dict[str, str]:
    env = os.environ.copy()
    root = _repo_root(here)
    prepend = [str(root / "build" / "Release"), str(here.parent)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        prepend.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(prepend)
    return env


def _run_python(
    py: str,
    here: Path,
    method: str,
    params: List[Union[float, str]],
    max_iter: int,
    dt_analysis: float,
    scale: float,
    gm_file: Path,
    pflag: Optional[int],
    system: Optional[str],
) -> int:
    cmd = [
        py,
        str(here / "run_one_integrator.py"),
        "--method",
        method,
        "--params",
        json.dumps(params),
        "--maxIter",
        str(max_iter),
        "--gm",
        str(gm_file),
        "--dt_analysis",
        str(dt_analysis),
        "--scale",
        str(scale),
    ]
    if pflag is not None:
        cmd += ["--pFlag", str(pflag)]
    if system is not None:
        cmd += ["--system", system]
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(here), env=_python_env(here)).returncode


RunTask = Tuple[
    str, str, Path, str, List[Union[float, str]], int, float, float, Path, Optional[int], Optional[str]
]


def _run_task(task: RunTask) -> Tuple[str, int]:
    label, py, here, method, params, max_iter, dt_analysis, scale, gm_file, pflag, system = task
    rc = _run_python(
        py, here, method, params, max_iter, dt_analysis, scale, gm_file, pflag, system
    )
    return label, rc


def _kr_cpu_run(rho: float) -> RunSpec:
    return (f"KR ρ={rho:g}", "KRAlphaExplicit", [rho], None)


def _mkr_cpu_run(rho: float) -> RunSpec:
    return (f"MKR ρ={rho:g}", "MKRAlphaExplicit", [rho], None)


def _cuda_runs(
    rho: float, *, incremental: bool = False, alpha_close_check: bool = False
) -> List[RunSpec]:
    sfx = ""
    extra: List[Union[float, str]] = []
    if incremental:
        sfx += " (incr)"
        extra.append("-incrementalAccel")
    if alpha_close_check:
        sfx += " (α close)"
        extra.append("-alphaCloseCheck")
    p: List[Union[float, str]] = [rho, *extra]
    return [
        (f"CudaKR ρ={rho:g}{sfx}", "CudaKRAlpha", list(p), None),
        (f"CudaMKR ρ={rho:g}{sfx}", "CudaMKRAlpha", list(p), None),
    ]


def _multisoe_runs(
    rho: float, *, incremental: bool = False, alpha_close_check: bool = False
) -> List[RunSpec]:
    sfx = ""
    extra: List[Union[float, str]] = []
    if incremental:
        sfx += " (incr)"
        extra.append("-incrementalAccel")
    if alpha_close_check:
        sfx += " (α close)"
        extra.append("-alphaCloseCheck")
    p: List[Union[float, str]] = [rho, *extra]
    return [
        (f"MultiSOE KR ρ={rho:g}{sfx}", "KRAlphaExplicitMultiSOE", list(p), None),
        (f"MultiSOE MKR ρ={rho:g}{sfx}", "MKRAlphaExplicitMultiSOE", list(p), None),
    ]


def build_runs(rhos: List[float], *, include_incremental: bool = True) -> List[RunSpec]:
    runs: List[RunSpec] = [
        ("Newmark (CuDSS)", "Newmark", [0.5, 0.25], None),
        ("Newmark (CPU)", "Newmark", [0.5, 0.25], "FullGeneral"),
    ]
    for rho in rhos:
        runs.append(_kr_cpu_run(rho))
        runs.append(_mkr_cpu_run(rho))
        runs.extend(_multisoe_runs(rho, incremental=False))
        runs.extend(_cuda_runs(rho, incremental=False))
        if include_incremental:
            runs.extend(_multisoe_runs(rho, incremental=True))
            runs.extend(_cuda_runs(rho, incremental=True))
        if abs(rho - 1.0) < 1e-12:
            runs.extend(_multisoe_runs(rho, alpha_close_check=True))
            runs.extend(_cuda_runs(rho, alpha_close_check=True))
            if include_incremental:
                runs.extend(
                    _multisoe_runs(rho, incremental=True, alpha_close_check=True)
                )
                runs.extend(
                    _cuda_runs(rho, incremental=True, alpha_close_check=True)
                )
    return runs


_SKIP_ARGS = frozenset(("--plots-only", "--append", "--no-incremental"))


def _parse_rho_args(argv: List[str]) -> List[float]:
    rhos: List[float] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in _SKIP_ARGS:
            i += 1
            continue
        try:
            rhos.append(float(arg))
        except ValueError:
            pass
        i += 1
    return rhos


def _run_all_tasks(tasks: List[RunTask]) -> None:
    print(f"Running {len(tasks)} shear analyses (serial)", flush=True)
    for task in tasks:
        label, rc = _run_task(task)
        if rc != 0:
            print(f"WARNING: {label} failed (exit code {rc})", flush=True)


def main(here: Path) -> None:
    os.chdir(here)
    argv = sys.argv[1:]

    if "--jobs" in argv or "-j" in argv:
        print("ERROR: --jobs was removed; analyses always run serially", file=sys.stderr)
        raise SystemExit(2)

    plots_only = "--plots-only" in argv
    append = "--append" in argv
    only_rhos = _parse_rho_args(argv)
    include_incremental = "--no-incremental" not in argv
    rhos = only_rhos if only_rhos else DEFAULT_RHOS
    runs = build_runs(rhos, include_incremental=include_incremental)

    if not plots_only:
        import importlib.util

        cfg_path = here / "plot_config.py"
        spec = importlib.util.spec_from_file_location("plot_config", cfg_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {cfg_path}")
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)

        gm_file = here / "tabasFN.txt"
        if not gm_file.is_file():
            print(f"ERROR: ground motion not found: {gm_file}", file=sys.stderr)
            raise SystemExit(2)

        py = os.environ.get("PYTHON", "python3")
        dt_analysis = float(getattr(cfg, "DT_ANALYSIS", 0.02))
        gm_scale = float(getattr(cfg, "GM_SCALE", 9.81))

        if not append:
            for sub in ("results", "figures"):
                d = here / sub
                if d.is_dir():
                    shutil.rmtree(d)
                    print(f"Removed {d}")

        tasks: List[RunTask] = []
        for label, ops_method, params, system in runs:
            max_iter = 25 if ops_method == "Newmark" else 1
            pflag = 0 if ops_method == "Newmark" else 5
            tasks.append(
                (
                    label,
                    py,
                    here,
                    ops_method,
                    params,
                    max_iter,
                    dt_analysis,
                    gm_scale,
                    gm_file,
                    pflag,
                    system,
                )
            )
        _run_all_tasks(tasks)

    sys.path.insert(0, str(here.parent))
    from plotResults import run as plot_results

    rc = plot_results(here)
    if rc != 0:
        raise SystemExit(rc)

    from collect_timing import write_timing_summary

    write_timing_summary(here)
