# -*- coding: utf-8 -*-
"""
Run the Two-Story MRF integrator matrix, then plot with Python.

Analyses run in separate subprocesses (Tcl OpenSees or OpenSeesPy) so a crash in
one case does not block plotting the others. Prefer ``--engine tcl`` to avoid the
known Python teardown segfault in the local ``opensees.so``.

Each CUDA/CPU KR family also runs with ``-incrementalAccel``; at rho=1.0 add
``-alphaCloseCheck`` (and both flags together), matching the KRAlphaSparse layout.
``-incrementalAccel`` and ``-alphaCloseCheck`` apply to **CudaKRAlpha / CudaMKRAlpha /
KRAlphaExplicitMultiSOE / MKRAlphaExplicitMultiSOE**; dense CPU ``KRAlphaExplicit`` /
``MKRAlphaExplicit`` are always the standard (total-form) case.

Usage:
  python3 run_integrators.py                    # Tcl, rhos 1.0 and 0.5
  python3 run_integrators.py --engine python    # OpenSeesPy via run_one_integrator.py
  python3 run_integrators.py --engine tcl        # explicit Tcl (default)
  python3 run_integrators.py 0.75
  python3 run_integrators.py --append 1.0
  python3 run_integrators.py --plots-only
  python3 run_integrators.py --no-incremental
  python3 run_integrators.py --cudss-dffi --append 1.0
  # dFFI matrix includes IR=0 (default), IR=2, and IR=5 variants
  python3 run_integrators.py --jobs auto   # fast wall clock; CUDA timings inflated on 1 GPU
  python3 run_integrators.py --jobs 1      # serial: accurate per-case wall_time_s in timing.txt

Environment:
  OPENSEES   path to OpenSees executable (Tcl; default: ../../../build/Release/OpenSees)
  PYTHON     Python executable for --engine python (default: python3)
  PYTHONPATH prepended with build/Release and EXAMPLES/KRAlphaExplicit for Python runs
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

Engine = Literal["tcl", "python"]
RunSpec = Tuple[str, str, List[Union[float, str]], Optional[str], Optional[str], int]

from plot_config import DEFAULT_RHOS

DEFAULT_SCALE = 3.0
DEFAULT_ENGINE: Engine = "tcl"
CUDSS_DFFI_IR_STEPS: List[int] = [0, 2, 5]


def _repo_root(here: Path) -> Path:
    return (here / ".." / ".." / "..").resolve()


def _default_opensees(here: Path) -> Path:
    env = os.environ.get("OPENSEES")
    if env:
        return Path(env)
    return _repo_root(here) / "build" / "Release" / "OpenSees"


def _python_env(here: Path) -> dict[str, str]:
    env = os.environ.copy()
    root = _repo_root(here)
    prepend = [str(root / "build" / "Release"), str(here.parent)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        prepend.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(prepend)
    return env


def _tcl_method_and_args(
    ops_method: str,
    params: List[Union[float, str]],
    scale: float,
    system: Optional[str],
    cudss_precision: Optional[str] = None,
    cudss_ir_n_steps: int = 0,
) -> List[str]:
    if ops_method == "Newmark" and system == "FullGeneral":
        return ["NewmarkCPU"]
    if ops_method == "Newmark":
        args: List[str] = ["Newmark"]
        if system is not None and system != "CuDSS":
            args.extend(["-system", system])
        if cudss_precision is not None:
            args.extend(["-cudssPrecision", cudss_precision])
        if cudss_ir_n_steps > 0:
            args.extend(["-cudssIrNSteps", str(cudss_ir_n_steps)])
        return args

    if not params:
        raise ValueError(f"integrator {ops_method} requires params")

    args = [ops_method, str(params[0]), str(scale)]
    for tok in params[1:]:
        if isinstance(tok, str):
            if not tok.startswith("-"):
                raise ValueError(f"unexpected integrator flag token: {tok!r}")
            args.append(tok)
        else:
            raise ValueError(f"unexpected integrator param after rho: {tok!r}")
    if system is not None and system != "CuDSS":
        args.extend(["-system", system])
    if cudss_precision is not None:
        args.extend(["-cudssPrecision", cudss_precision])
    if cudss_ir_n_steps > 0:
        args.extend(["-cudssIrNSteps", str(cudss_ir_n_steps)])
    return args


def _run_tcl(
    opensees: Path,
    here: Path,
    method: str,
    params: List[Union[float, str]],
    scale: float,
    system: Optional[str],
    cudss_precision: Optional[str],
    cudss_ir_n_steps: int,
) -> int:
    tcl_script = here / "two_story_MRF.tcl"
    cmd = [
        str(opensees),
        str(tcl_script),
        *_tcl_method_and_args(method, params, scale, system, cudss_precision, cudss_ir_n_steps),
    ]
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=here).returncode


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
    cudss_precision: Optional[str],
    cudss_ir_n_steps: int,
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
    if cudss_precision is not None:
        cmd += ["--cudss-precision", cudss_precision]
    if cudss_ir_n_steps > 0:
        cmd += ["--cudss-ir-n-steps", str(cudss_ir_n_steps)]
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=here, env=_python_env(here)).returncode


RunTask = Tuple[
    str,  # label
    Engine,
    Path,  # here
    str,  # method
    List[Union[float, str]],
    float,  # scale
    Optional[str],  # system
    Optional[str],  # cudss_precision
    int,  # cudss_ir_n_steps
    Path,  # gm_file
    float,  # dt_analysis
    Optional[Path],  # opensees (tcl)
    str,  # python executable
]


def _run_task(task: RunTask) -> Tuple[str, int]:
    (
        label,
        engine,
        here,
        method,
        params,
        scale,
        system,
        cudss_precision,
        cudss_ir_n_steps,
        gm_file,
        dt_analysis,
        opensees,
        py,
    ) = task
    max_iter = 25 if method == "Newmark" else 1
    pflag = 0 if method == "Newmark" else 5
    if engine == "tcl":
        rc = _run_tcl(opensees, here, method, params, scale, system, cudss_precision, cudss_ir_n_steps)
    else:
        rc = _run_python(
            py,
            here,
            method,
            params,
            max_iter,
            dt_analysis,
            scale,
            gm_file,
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
    system: Optional[str] = None,
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
            system,
            cudss_precision,
            cudss_ir_n_steps,
        ),
        (
            f"MultiSOE MKR ρ={rho:g}{sfx}",
            "MKRAlphaExplicitMultiSOE",
            list(p),
            system,
            cudss_precision,
            cudss_ir_n_steps,
        ),
    ]


def _build_soe_variant_runs(rhos: List[float], *, include_incremental: bool = True) -> List[RunSpec]:
    """UmfPack/SuperLU Newmark + MultiSOE only (append to existing CuDSS matrix)."""
    runs: List[RunSpec] = [
        ("Newmark (UmfPack)", "Newmark", [0.5, 0.25], "UmfPack", None, 0),
        ("Newmark (SuperLU)", "Newmark", [0.5, 0.25], "SuperLU", None, 0),
    ]
    for rho in rhos:
        runs.extend(_multisoe_runs(rho, incremental=False, system="UmfPack", label="UmfPack"))
        runs.extend(_multisoe_runs(rho, incremental=False, system="SuperLU", label="SuperLU"))
        if include_incremental:
            runs.extend(_multisoe_runs(rho, incremental=True, system="UmfPack", label="UmfPack"))
            runs.extend(_multisoe_runs(rho, incremental=True, system="SuperLU", label="SuperLU"))
        if abs(rho - 1.0) < 1e-12:
            runs.extend(
                _multisoe_runs(rho, alpha_close_check=True, system="UmfPack", label="UmfPack")
            )
            runs.extend(
                _multisoe_runs(rho, alpha_close_check=True, system="SuperLU", label="SuperLU")
            )
            if include_incremental:
                runs.extend(
                    _multisoe_runs(
                        rho,
                        incremental=True,
                        alpha_close_check=True,
                        system="UmfPack",
                        label="UmfPack",
                    )
                )
                runs.extend(
                    _multisoe_runs(
                        rho,
                        incremental=True,
                        alpha_close_check=True,
                        system="SuperLU",
                        label="SuperLU",
                    )
                )
    return runs


def _cudss_sp_label(ir_n_steps: int) -> str:
    if ir_n_steps <= 0:
        return "CuDSS sp"
    return f"CuDSS sp IR={ir_n_steps}"


def _build_cudss_sp_runs(rhos: List[float], *, include_incremental: bool = True) -> List[RunSpec]:
    """CuDSS single-precision (dFFI) for all CuDSS code paths (append mode)."""
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


def _build_runs(rhos: List[float], *, include_incremental: bool = True) -> List[RunSpec]:
    runs: List[RunSpec] = [
        ("Newmark (CuDSS)", "Newmark", [0.5, 0.25], None, None, 0),
        ("Newmark (CPU)", "Newmark", [0.5, 0.25], "FullGeneral", None, 0),
        ("Newmark (UmfPack)", "Newmark", [0.5, 0.25], "UmfPack", None, 0),
        ("Newmark (SuperLU)", "Newmark", [0.5, 0.25], "SuperLU", None, 0),
    ]
    for rho in rhos:
        runs.append(_kr_cpu_run(rho))
        runs.append(_mkr_cpu_run(rho))
        runs.extend(_multisoe_runs(rho, incremental=False))
        runs.extend(_cuda_runs(rho, incremental=False))
        runs.extend(_multisoe_runs(rho, incremental=False, system="UmfPack", label="UmfPack"))
        runs.extend(_multisoe_runs(rho, incremental=False, system="SuperLU", label="SuperLU"))
        if include_incremental:
            runs.extend(_multisoe_runs(rho, incremental=True))
            runs.extend(_cuda_runs(rho, incremental=True))
            runs.extend(_multisoe_runs(rho, incremental=True, system="UmfPack", label="UmfPack"))
            runs.extend(_multisoe_runs(rho, incremental=True, system="SuperLU", label="SuperLU"))
        if abs(rho - 1.0) < 1e-12:
            runs.extend(_multisoe_runs(rho, alpha_close_check=True))
            runs.extend(_cuda_runs(rho, alpha_close_check=True))
            runs.extend(
                _multisoe_runs(rho, alpha_close_check=True, system="UmfPack", label="UmfPack")
            )
            runs.extend(
                _multisoe_runs(rho, alpha_close_check=True, system="SuperLU", label="SuperLU")
            )
            if include_incremental:
                runs.extend(
                    _multisoe_runs(rho, incremental=True, alpha_close_check=True)
                )
                runs.extend(
                    _cuda_runs(rho, incremental=True, alpha_close_check=True)
                )
                runs.extend(
                    _multisoe_runs(
                        rho,
                        incremental=True,
                        alpha_close_check=True,
                        system="UmfPack",
                        label="UmfPack",
                    )
                )
                runs.extend(
                    _multisoe_runs(
                        rho,
                        incremental=True,
                        alpha_close_check=True,
                        system="SuperLU",
                        label="SuperLU",
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
        "--engine",
        "--tcl",
        "--python",
        "--soe-variants",
        "--cudss-sp",
        "--cudss-dffi",
    )
)


def _parse_engine(argv: List[str]) -> Engine:
    if "--python" in argv:
        return "python"
    if "--tcl" in argv:
        return "tcl"
    for i, arg in enumerate(argv):
        if arg == "--engine" and i + 1 < len(argv):
            val = argv[i + 1].lower()
            if val in ("tcl", "python"):
                return val  # type: ignore[return-value]
            print(f"ERROR: unknown --engine {argv[i + 1]!r} (want tcl or python)", file=sys.stderr)
            raise SystemExit(2)
    return DEFAULT_ENGINE


def _parse_rho_args(argv: List[str]) -> List[float]:
    rhos: List[float] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in _SKIP_ARGS:
            i += 2 if arg in ("--jobs", "-j", "--engine") else 1
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


def _run_all_tasks(tasks: List[RunTask], jobs: int, engine: Engine) -> None:
    if jobs <= 1 or len(tasks) <= 1:
        for task in tasks:
            label, rc = _run_task(task)
            if rc != 0:
                print(f"WARNING: {label} failed (exit code {rc})", flush=True)
        return

    print(f"Running {len(tasks)} {engine} analyses with {jobs} workers", flush=True)
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
    soe_variants_only = "--soe-variants" in argv
    cudss_sp_only = "--cudss-sp" in argv or "--cudss-dffi" in argv
    jobs = _parse_jobs(argv)
    engine = _parse_engine(argv)
    only_rhos = _parse_rho_args(argv)
    include_incremental = "--no-incremental" not in argv
    rhos = only_rhos if only_rhos else DEFAULT_RHOS
    if cudss_sp_only:
        runs = _build_cudss_sp_runs(rhos, include_incremental=include_incremental)
    elif soe_variants_only:
        runs = _build_soe_variant_runs(rhos, include_incremental=include_incremental)
    else:
        runs = _build_runs(rhos, include_incremental=include_incremental)

    if not plots_only:
        from plot_config import DT_ANALYSIS

        gm_file = here / "ground_motions" / "RSN960_NORTHR_LOS270.AT2"
        py = os.environ.get("PYTHON", "python3")
        opensees = _default_opensees(here)

        if engine == "tcl" and not opensees.is_file():
            print(f"ERROR: OpenSees executable not found: {opensees}", file=sys.stderr)
            print("Set OPENSEES or build build/Release/OpenSees", file=sys.stderr)
            raise SystemExit(2)

        if not append:
            for sub in ("results", "figures"):
                d = here / sub
                if d.is_dir():
                    shutil.rmtree(d)
                    print(f"Removed {d}")

        print(f"Analysis engine: {engine}", flush=True)
        tasks: List[RunTask] = []
        for label, ops_method, params, system, cudss_precision, cudss_ir_n_steps in runs:
            tasks.append(
                (
                    label,
                    engine,
                    here,
                    ops_method,
                    params,
                    DEFAULT_SCALE,
                    system,
                    cudss_precision,
                    cudss_ir_n_steps,
                    gm_file,
                    float(DT_ANALYSIS),
                    opensees,
                    py,
                )
            )
        _run_all_tasks(tasks, jobs, engine)

    sys.path.insert(0, str(here.parent))
    from plotResults import run as plot_results

    rc = plot_results(here, jobs=jobs)
    if rc != 0:
        raise SystemExit(rc)

    from collect_timing import write_timing_summary

    write_timing_summary(here)


if __name__ == "__main__":
    main()
