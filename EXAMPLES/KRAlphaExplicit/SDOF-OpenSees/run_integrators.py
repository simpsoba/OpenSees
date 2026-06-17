# -*- coding: utf-8 -*-
"""
Run the SDOF integrator matrix (same layout as Two-Story_MRF on ops-cuda), then plot vs theory.

Each case uses initial conditions from ``plot_config.IC_CASES`` and writes under
``results/<ic>/<dt_tag>/<integrator>_params-.../`` for each ``DT_CASES`` entry.

By default, ``results/`` and ``figures/`` are removed before each run (use ``--append`` to keep).

Usage:
  python3 run_integrators.py
  python3 run_integrators.py 0.75 1.0
  python3 run_integrators.py --append 0.5
  python3 run_integrators.py --plots-only
  python3 run_integrators.py --jobs auto
  python3 run_integrators.py --no-incremental
  python3 run_integrators.py --cudss-dffi --append
  # dFFI matrix includes IR=0 (default), IR=2, and IR=5 variants
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple, Union

RunSpec = Tuple[str, str, List[Union[float, str]], Optional[str], Optional[str], int]

from plot_config import DEFAULT_RHOS

CUDSS_DFFI_IR_STEPS: List[int] = [0, 2, 5]


def _repo_root(here: Path) -> Path:
    return (here / ".." / ".." / "..").resolve()


def _python_env(here: Path | str) -> dict[str, str]:
    here_path = Path(here)
    env = os.environ.copy()
    root = _repo_root(here_path)
    prepend = [str(root / "build" / "Release"), str(here.parent)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        prepend.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(prepend)
    return env


def _run_python(
    py: str,
    here: Path | str,
    ic_tag: str,
    method: str,
    params: List[Union[float, str]],
    max_iter: int,
    dt_analysis: float,
    dt_tag: str,
    pflag: Optional[int],
    system: Optional[str],
    cudss_precision: Optional[str],
    cudss_ir_n_steps: int,
) -> int:
    here_path = Path(here)
    cmd = [
        py,
        str(here_path / "run_one_integrator.py"),
        "--method",
        method,
        "--params",
        json.dumps(params),
        "--maxIter",
        str(max_iter),
        "--ic",
        ic_tag,
        "--dt_analysis",
        str(dt_analysis),
        "--dt_tag",
        dt_tag,
    ]
    if pflag is not None:
        cmd += ["--pFlag", str(pflag)]
    if system is not None:
        cmd += ["--system", system]
    if cudss_precision is not None:
        cmd += ["--cudss-precision", cudss_precision]
    if cudss_ir_n_steps > 0:
        cmd += ["--cudss-ir-n-steps", str(cudss_ir_n_steps)]
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(here_path), env=_python_env(here_path)).returncode


RunTask = Tuple[
    str,
    str,
    str,
    str,
    str,
    List[Union[float, str]],
    int,
    float,
    str,
    Optional[int],
    Optional[str],
    Optional[str],
    int,
]


def _run_task(task: RunTask) -> Tuple[str, int]:
    (
        label,
        py,
        here,
        ic_tag,
        method,
        params,
        max_iter,
        dt_analysis,
        dt_tag,
        pflag,
        system,
        cudss_precision,
        cudss_ir_n_steps,
    ) = task
    rc = _run_python(
        py,
        here,
        ic_tag,
        method,
        params,
        max_iter,
        dt_analysis,
        dt_tag,
        pflag,
        system,
        cudss_precision,
        cudss_ir_n_steps,
    )
    return label, rc


def _kr_cpu_run(rho: float) -> RunSpec:
    return (f"KR ρ={rho:g}", "KRAlphaExplicit", [rho], None, None, 0)


def _mkr_cpu_run(rho: float) -> RunSpec:
    return (f"MKR ρ={rho:g}", "MKRAlphaExplicit", [rho], None, None, 0)


def _cuda_runs(
    rho: float,
    *,
    incremental: bool = False,
    alpha_close_check: bool = False,
    cudss_precision: Optional[str] = None,
    cudss_ir_n_steps: int = 0,
    label: Optional[str] = None,
) -> List[RunSpec]:
    sfx = ""
    extra: List[Union[float, str]] = []
    if incremental:
        sfx += " (incr)"
        extra.append("-incrementalAccel")
    if alpha_close_check:
        sfx += " (α close)"
        extra.append("-alphaCloseCheck")
    if label:
        sfx += f" ({label})"
    p: List[Union[float, str]] = [rho, *extra]
    return [
        (f"CudaKR ρ={rho:g}{sfx}", "CudaKRAlpha", list(p), None, cudss_precision, cudss_ir_n_steps),
        (f"CudaMKR ρ={rho:g}{sfx}", "CudaMKRAlpha", list(p), None, cudss_precision, cudss_ir_n_steps),
    ]


def _multisoe_runs(
    rho: float,
    *,
    incremental: bool = False,
    alpha_close_check: bool = False,
    cudss_precision: Optional[str] = None,
    cudss_ir_n_steps: int = 0,
    label: Optional[str] = None,
) -> List[RunSpec]:
    sfx = ""
    extra: List[Union[float, str]] = []
    if incremental:
        sfx += " (incr)"
        extra.append("-incrementalAccel")
    if alpha_close_check:
        sfx += " (α close)"
        extra.append("-alphaCloseCheck")
    if label:
        sfx += f" ({label})"
    p: List[Union[float, str]] = [rho, *extra]
    return [
        (
            f"MultiSOE KR ρ={rho:g}{sfx}",
            "KRAlphaExplicitMultiSOE",
            list(p),
            None,
            cudss_precision,
            cudss_ir_n_steps,
        ),
        (
            f"MultiSOE MKR ρ={rho:g}{sfx}",
            "MKRAlphaExplicitMultiSOE",
            list(p),
            None,
            cudss_precision,
            cudss_ir_n_steps,
        ),
    ]


def _build_runs(rhos: List[float], *, include_incremental: bool = True) -> List[RunSpec]:
    runs: List[RunSpec] = [
        ("Newmark (CuDSS)", "Newmark", [0.5, 0.25], None, None, 0),
        ("Newmark (CPU)", "Newmark", [0.5, 0.25], "FullGeneral", None, 0),
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


def _cudss_sp_label(ir_n_steps: int) -> str:
    if ir_n_steps <= 0:
        return "CuDSS sp"
    return f"CuDSS sp IR={ir_n_steps}"


def _build_cudss_sp_runs(rhos: List[float], *, include_incremental: bool = True) -> List[RunSpec]:
    """CuDSS single-precision (dFFI) for all CuDSS code paths."""
    sp = "dFFI"
    runs: List[RunSpec] = []
    for ir_n_steps in CUDSS_DFFI_IR_STEPS:
        sp_label = _cudss_sp_label(ir_n_steps)
        ir_suffix = "" if ir_n_steps <= 0 else f" IR={ir_n_steps}"
        runs.append(
            (f"Newmark (CuDSS sp{ir_suffix})", "Newmark", [0.5, 0.25], None, sp, ir_n_steps)
        )
        for rho in rhos:
            runs.extend(
                _multisoe_runs(
                    rho,
                    incremental=False,
                    cudss_precision=sp,
                    cudss_ir_n_steps=ir_n_steps,
                    label=sp_label,
                )
            )
            runs.extend(
                _cuda_runs(
                    rho,
                    incremental=False,
                    cudss_precision=sp,
                    cudss_ir_n_steps=ir_n_steps,
                    label=sp_label,
                )
            )
            if include_incremental:
                runs.extend(
                    _multisoe_runs(
                        rho,
                        incremental=True,
                        cudss_precision=sp,
                        cudss_ir_n_steps=ir_n_steps,
                        label=sp_label,
                    )
                )
                runs.extend(
                    _cuda_runs(
                        rho,
                        incremental=True,
                        cudss_precision=sp,
                        cudss_ir_n_steps=ir_n_steps,
                        label=sp_label,
                    )
                )
            if abs(rho - 1.0) < 1e-12:
                runs.extend(
                    _multisoe_runs(
                        rho,
                        alpha_close_check=True,
                        cudss_precision=sp,
                        cudss_ir_n_steps=ir_n_steps,
                        label=sp_label,
                    )
                )
                runs.extend(
                    _cuda_runs(
                        rho,
                        alpha_close_check=True,
                        cudss_precision=sp,
                        cudss_ir_n_steps=ir_n_steps,
                        label=sp_label,
                    )
                )
                if include_incremental:
                    runs.extend(
                        _multisoe_runs(
                            rho,
                            incremental=True,
                            alpha_close_check=True,
                            cudss_precision=sp,
                            cudss_ir_n_steps=ir_n_steps,
                            label=sp_label,
                        )
                    )
                    runs.extend(
                        _cuda_runs(
                            rho,
                            incremental=True,
                            alpha_close_check=True,
                            cudss_precision=sp,
                            cudss_ir_n_steps=ir_n_steps,
                            label=sp_label,
                        )
                    )
    return runs


_SKIP_ARGS = frozenset(
    (
        "--plots-only",
        "--append",
        "--jobs",
        "-j",
        "--no-incremental",
        "--cudss-sp",
        "--cudss-dffi",
    )
)


def _parse_rho_args(argv: List[str]) -> List[float]:
    rhos: List[float] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in _SKIP_ARGS:
            i += 2 if arg in ("--jobs", "-j") else 1
            continue
        try:
            rhos.append(float(arg))
        except ValueError:
            pass
        i += 1
    return rhos


def _parse_jobs(argv: List[str]) -> int:
    for i, arg in enumerate(argv):
        if arg in ("--jobs", "-j") and i + 1 < len(argv):
            val = argv[i + 1]
            if val.lower() == "auto":
                return max(1, os.cpu_count() or 1)
            return max(1, int(val))
    return 1


def _run_all_tasks(tasks: List[RunTask], jobs: int) -> None:
    if jobs <= 1 or len(tasks) <= 1:
        for task in tasks:
            label, rc = _run_task(task)
            if rc != 0:
                print(f"WARNING: {label} failed (exit code {rc})", flush=True)
        return

    print(f"Running {len(tasks)} SDOF analyses with {jobs} workers", flush=True)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_run_task, task): task[0] for task in tasks}
        for fut in as_completed(futures):
            label = futures[fut]
            try:
                label, rc = fut.result()
            except Exception as exc:
                print(f"WARNING: {label} raised {exc!r}", flush=True)
                continue
            if rc != 0:
                print(f"WARNING: {label} failed (exit code {rc})", flush=True)


def main() -> None:
    here = Path(__file__).resolve().parent
    os.chdir(here)
    argv = sys.argv[1:]

    plots_only = "--plots-only" in argv
    append = "--append" in argv
    cudss_sp_only = "--cudss-sp" in argv or "--cudss-dffi" in argv
    jobs = _parse_jobs(argv)
    only_rhos = _parse_rho_args(argv)
    include_incremental = "--no-incremental" not in argv
    rhos = only_rhos if only_rhos else DEFAULT_RHOS
    if cudss_sp_only:
        runs = _build_cudss_sp_runs(rhos, include_incremental=include_incremental)
    else:
        runs = _build_runs(rhos, include_incremental=include_incremental)

    if not plots_only:
        from plot_config import DT_CASES, IC_CASES

        py = os.environ.get("PYTHON", "python3")

        if not append:
            for sub in ("results", "figures"):
                d = here / sub
                if d.is_dir():
                    shutil.rmtree(d)
                    print(f"Removed {d}")

        tasks: List[RunTask] = []
        for dt_case in DT_CASES:
            dt_analysis = float(dt_case["dt"])
            dt_tag = str(dt_case["tag"])
            for ic in IC_CASES:
                ic_tag = ic["tag"]
                for label, ops_method, params, system, cudss_precision, cudss_ir_n_steps in runs:
                    max_iter = 25 if ops_method == "Newmark" else 1
                    pflag = 0 if ops_method == "Newmark" else 5
                    tasks.append(
                        (
                            f"{label} | {ic_tag} | {dt_tag}",
                            py,
                            str(here),
                            ic_tag,
                            ops_method,
                            params,
                            max_iter,
                            dt_analysis,
                            dt_tag,
                            pflag,
                            system,
                            cudss_precision,
                            cudss_ir_n_steps,
                        )
                    )
        _run_all_tasks(tasks, jobs)

    from plotResults import run as plot_results

    rc = plot_results(here, jobs=jobs)
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
