# -*- coding: utf-8 -*-
"""
Run the Two-Story MRF integrator matrix, then plot with Python.

Analyses run in separate subprocesses (Tcl OpenSees or OpenSeesPy) so a crash in
one case does not block plotting the others. Prefer ``--engine tcl`` to avoid the
known Python teardown segfault in the local ``opensees.so``.

Run matrix (same layout at every ρ passed via ``--rho``):
  Default — full double-precision matrix (Newmark baselines + KR/MKR CPU, MultiSOE,
  CUDA, UmfPack/SuperLU, standard  ``-incrementalAccel`` variants).
  ``--cudss-dffi-only`` — CuDSS single-precision (dFFI) subset only (IR=0,2,5).
  ``--alpha-close-check-only`` — ``-alphaCloseCheck`` subset only.
  ``--numberer Plain|RCM|AMD`` — DOF numberer for this pass (default RCM).
  ``--reuse-newmark`` — with ``--append``, skip baseline Newmark runs (keep dFFI Newmark).
  ``--soe-variants`` — UmfPack/SuperLU Newmark + MultiSOE append pass.

``-incrementalAccel`` and ``-alphaCloseCheck`` apply to **CudaKRAlpha / CudaMKRAlpha /
KRAlphaExplicitMultiSOE / MKRAlphaExplicitMultiSOE** only; dense CPU KR/MKR are always
the standard (total-form) case. Use ``--append`` on modifier passes to accumulate results.

Usage:
  python3 run_integrators.py --engine tcl --rho 0.5 0.9 1.0 --all-mass-modes
  python3 run_integrators.py --alpha-close-check-only --append --engine tcl --rho 1.0 --all-mass-modes
  python3 run_integrators.py --cudss-dffi-only --append --engine tcl --rho 0.5 0.9 1.0 --all-mass-modes
  python3 run_integrators.py --engine tcl --rho 1.0 --mass-mode 2
  python3 run_integrators.py --numberer Plain --append --reuse-newmark --engine tcl --rho 1.0 --mass-mode 0

Analyses always run serially (one subprocess at a time) for reliable CuDSS/CUDA
timing and to avoid GPU contention.

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
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_utils import VALID_NUMBERERS
from plot_config import DEFAULT_RHOS, output_dirs

Engine = Literal["tcl", "python"]
RunSpec = Tuple[str, str, List[Union[float, str]], Optional[str], Optional[str], int]

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


def _parse_numberer(argv: List[str]) -> str:
    for i, arg in enumerate(argv):
        if arg == "--numberer" and i + 1 < len(argv):
            val = argv[i + 1]
            if val not in VALID_NUMBERERS:
                print(
                    f"ERROR: invalid --numberer {val!r} (want Plain, RCM, or AMD)",
                    file=sys.stderr,
                )
                raise SystemExit(2)
            return val
    return "RCM"


def _is_skippable_baseline_newmark(run: RunSpec) -> bool:
    """True for standard Newmark baselines that need not be re-run on append passes."""
    _label, method, _params, _system, cudss_precision, _ir = run
    if method != "Newmark":
        return False
    return cudss_precision != "dFFI"


def _filter_reuse_newmark(runs: List[RunSpec], *, reuse_newmark: bool) -> List[RunSpec]:
    if not reuse_newmark:
        return runs
    kept: List[RunSpec] = []
    for run in runs:
        if _is_skippable_baseline_newmark(run):
            print(f"SKIP reuse-newmark: {run[0]}", flush=True)
            continue
        kept.append(run)
    return kept


def _parse_mass_mode(argv: List[str]) -> int:
    for i, arg in enumerate(argv):
        if arg == "--mass-mode" and i + 1 < len(argv):
            try:
                mode = int(argv[i + 1])
            except ValueError:
                print(f"ERROR: invalid --mass-mode {argv[i + 1]!r}", file=sys.stderr)
                raise SystemExit(2)
            if mode not in (0, 1, 2):
                print("ERROR: --mass-mode must be 0, 1, or 2", file=sys.stderr)
                raise SystemExit(2)
            return mode
    return 0


def _run_tcl(
    opensees: Path,
    here: Path,
    method: str,
    params: List[Union[float, str]],
    scale: float,
    system: Optional[str],
    cudss_precision: Optional[str],
    cudss_ir_n_steps: int,
    *,
    mass_mode: int = 0,
    numberer: str = "RCM",
) -> int:
    tcl_script = here / "two_story_MRF.tcl"
    cmd = [
        str(opensees),
        str(tcl_script),
        *_tcl_method_and_args(method, params, scale, system, cudss_precision, cudss_ir_n_steps),
    ]
    if mass_mode != 0:
        cmd.extend(["-massMode", str(mass_mode)])
    if numberer != "RCM":
        cmd.extend(["-numberer", numberer])
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
    *,
    mass_mode: int = 0,
    numberer: str = "RCM",
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
    if mass_mode != 0:
        cmd += ["--mass-mode", str(mass_mode)]
    if numberer != "RCM":
        cmd += ["--numberer", numberer]
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
    int,  # mass_mode
    str,  # numberer
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
        mass_mode,
        numberer,
    ) = task
    max_iter = 25 if method == "Newmark" else 1
    pflag = 0 if method == "Newmark" else 5
    if engine == "tcl":
        rc = _run_tcl(
            opensees,
            here,
            method,
            params,
            scale,
            system,
            cudss_precision,
            cudss_ir_n_steps,
            mass_mode=mass_mode,
            numberer=numberer,
        )
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
            mass_mode=mass_mode,
            numberer=numberer,
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


def _build_soe_variant_runs(
    rhos: List[float],
    *,
    include_incremental: bool = True,
) -> List[RunSpec]:
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
    return runs


def _cudss_dffi_label(ir_n_steps: int) -> str:
    if ir_n_steps <= 0:
        return "CuDSS dFFI"
    return f"CuDSS dFFI IR={ir_n_steps}"


def _build_cudss_dffi_runs(
    rhos: List[float],
    *,
    include_incremental: bool = True,
) -> List[RunSpec]:
    """CuDSS single-precision (dFFI) for all CuDSS code paths (append mode)."""
    precision = "dFFI"
    runs: List[RunSpec] = []
    for ir_n_steps in CUDSS_DFFI_IR_STEPS:
        dffi_label = _cudss_dffi_label(ir_n_steps)
        ir_suffix = "" if ir_n_steps <= 0 else f" IR={ir_n_steps}"
        runs.append(
            (f"Newmark (CuDSS dFFI{ir_suffix})", "Newmark", [0.5, 0.25], None, precision, ir_n_steps)
        )
        for rho in rhos:
            runs.extend(
                _multisoe_runs(
                    rho,
                    incremental=False,
                    cudss_precision=precision,
                    cudss_ir_n_steps=ir_n_steps,
                    label=dffi_label,
                )
            )
            runs.extend(
                _cuda_runs(
                    rho,
                    incremental=False,
                    cudss_precision=precision,
                    cudss_ir_n_steps=ir_n_steps,
                    label=dffi_label,
                )
            )
            if include_incremental:
                runs.extend(
                    _multisoe_runs(
                        rho,
                        incremental=True,
                        cudss_precision=precision,
                        cudss_ir_n_steps=ir_n_steps,
                        label=dffi_label,
                    )
                )
                runs.extend(
                    _cuda_runs(
                        rho,
                        incremental=True,
                        cudss_precision=precision,
                        cudss_ir_n_steps=ir_n_steps,
                        label=dffi_label,
                    )
                )
    return runs


def _append_alpha_close_check_runs(
    runs: List[RunSpec],
    rho: float,
    *,
    include_incremental: bool,
    cudss_precision: Optional[str] = None,
    cudss_ir_n_steps: int = 0,
    dffi_label: Optional[str] = None,
) -> None:
    runs.extend(
        _multisoe_runs(
            rho,
            alpha_close_check=True,
            cudss_precision=cudss_precision,
            cudss_ir_n_steps=cudss_ir_n_steps,
            label=dffi_label,
        )
    )
    runs.extend(
        _cuda_runs(
            rho,
            alpha_close_check=True,
            cudss_precision=cudss_precision,
            cudss_ir_n_steps=cudss_ir_n_steps,
            label=dffi_label,
        )
    )
    runs.extend(
        _multisoe_runs(
            rho,
            alpha_close_check=True,
            system="UmfPack",
            label="UmfPack" if dffi_label is None else dffi_label,
            cudss_precision=cudss_precision,
            cudss_ir_n_steps=cudss_ir_n_steps,
        )
    )
    runs.extend(
        _multisoe_runs(
            rho,
            alpha_close_check=True,
            system="SuperLU",
            label="SuperLU" if dffi_label is None else dffi_label,
            cudss_precision=cudss_precision,
            cudss_ir_n_steps=cudss_ir_n_steps,
        )
    )
    if include_incremental:
        runs.extend(
            _multisoe_runs(
                rho,
                incremental=True,
                alpha_close_check=True,
                cudss_precision=cudss_precision,
                cudss_ir_n_steps=cudss_ir_n_steps,
                label=dffi_label,
            )
        )
        runs.extend(
            _cuda_runs(
                rho,
                incremental=True,
                alpha_close_check=True,
                cudss_precision=cudss_precision,
                cudss_ir_n_steps=cudss_ir_n_steps,
                label=dffi_label,
            )
        )
        runs.extend(
            _multisoe_runs(
                rho,
                incremental=True,
                alpha_close_check=True,
                system="UmfPack",
                label="UmfPack" if dffi_label is None else dffi_label,
                cudss_precision=cudss_precision,
                cudss_ir_n_steps=cudss_ir_n_steps,
            )
        )
        runs.extend(
            _multisoe_runs(
                rho,
                incremental=True,
                alpha_close_check=True,
                system="SuperLU",
                label="SuperLU" if dffi_label is None else dffi_label,
                cudss_precision=cudss_precision,
                cudss_ir_n_steps=cudss_ir_n_steps,
            )
        )


def _build_alpha_close_check_runs(
    rhos: List[float],
    *,
    include_incremental: bool = True,
    cudss_dffi: bool = False,
) -> List[RunSpec]:
    """-alphaCloseCheck variants only (append after main matrix; any ρ in ``rhos``)."""
    runs: List[RunSpec] = []
    if cudss_dffi:
        precision = "dFFI"
        for ir_n_steps in CUDSS_DFFI_IR_STEPS:
            dffi_label = _cudss_dffi_label(ir_n_steps)
            for rho in rhos:
                _append_alpha_close_check_runs(
                    runs,
                    rho,
                    include_incremental=include_incremental,
                    cudss_precision=precision,
                    cudss_ir_n_steps=ir_n_steps,
                    dffi_label=dffi_label,
                )
        return runs

    for rho in rhos:
        _append_alpha_close_check_runs(runs, rho, include_incremental=include_incremental)
    return runs


def _build_runs(
    rhos: List[float],
    *,
    include_incremental: bool = True,
) -> List[RunSpec]:
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
    return runs


def _resolve_rhos(explicit_rhos: List[float]) -> List[float]:
    return list(explicit_rhos) if explicit_rhos else list(DEFAULT_RHOS)


def _build_run_list(
    rhos: List[float],
    *,
    cudss_dffi_only: bool,
    alpha_close_check_only: bool,
    soe_variants_only: bool,
    include_incremental: bool,
) -> List[RunSpec]:
    """Pick run matrix; every ρ in ``rhos`` gets the same case layout."""
    if cudss_dffi_only and alpha_close_check_only:
        return _build_alpha_close_check_runs(
            rhos,
            include_incremental=include_incremental,
            cudss_dffi=True,
        )
    if alpha_close_check_only:
        return _build_alpha_close_check_runs(rhos, include_incremental=include_incremental)
    if cudss_dffi_only:
        return _build_cudss_dffi_runs(rhos, include_incremental=include_incremental)
    if soe_variants_only:
        return _build_soe_variant_runs(rhos, include_incremental=include_incremental)
    return _build_runs(rhos, include_incremental=include_incremental)


def _parse_engine(argv: List[str]) -> Engine:
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
        if argv[i] != "--rho":
            i += 1
            continue
        i += 1
        if i >= len(argv) or argv[i].startswith("-"):
            print("ERROR: --rho requires at least one value", file=sys.stderr)
            raise SystemExit(2)
        while i < len(argv) and not argv[i].startswith("-"):
            try:
                rhos.append(float(argv[i]))
            except ValueError:
                print(f"ERROR: invalid --rho value {argv[i]!r}", file=sys.stderr)
                raise SystemExit(2)
            i += 1
    return rhos


def _run_all_tasks(tasks: List[RunTask], engine: Engine) -> None:
    print(f"Running {len(tasks)} {engine} analyses (serial)", flush=True)
    for task in tasks:
        label, rc = _run_task(task)
        if rc != 0:
            print(f"WARNING: {label} failed (exit code {rc})", flush=True)


def _run_mass_mode_matrix(
    here: Path,
    *,
    mass_mode: int,
    runs: List[RunSpec],
    engine: Engine,
    append: bool,
    plots_only: bool,
    figure_names: set[str] | None = None,
    numberer: str = "RCM",
) -> int:
    results_subdir, figures_subdir = output_dirs(mass_mode)

    if not plots_only:
        from plot_config import DT_ANALYSIS

        gm_file = here / "ground_motions" / "RSN960_NORTHR_LOS270.AT2"
        py = os.environ.get("PYTHON", "python3")
        opensees = _default_opensees(here)

        if engine == "tcl" and not opensees.is_file():
            print(f"ERROR: OpenSees executable not found: {opensees}", file=sys.stderr)
            print("Set OPENSEES or build build/Release/OpenSees", file=sys.stderr)
            return 2

        if not append:
            for sub in (results_subdir, figures_subdir):
                d = here / sub
                if d.is_dir():
                    shutil.rmtree(d)
                    print(f"Removed {d}")

        print(f"Analysis engine: {engine}", flush=True)
        print(f"Output: {results_subdir}/ , {figures_subdir}/", flush=True)
        if mass_mode != 0:
            print(f"Model: massMode={mass_mode} (Cuda adds -diagonalMass)", flush=True)
        if numberer != "RCM":
            print(f"DOF numberer: {numberer}", flush=True)
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
                    mass_mode,
                    numberer,
                )
            )
        _run_all_tasks(tasks, engine)

    sys.path.insert(0, str(here.parent))
    from plotResults import run as plot_results

    rc = plot_results(
        here,
        results_subdir=results_subdir,
        figures_subdir=figures_subdir,
        figure_names=figure_names,
    )
    if rc != 0:
        return rc

    from collect_timing import write_timing_summary

    write_timing_summary(here, results_subdir=results_subdir)
    return 0


def main() -> None:
    here = Path(__file__).resolve().parent
    os.chdir(here)
    argv = sys.argv[1:]

    if "--jobs" in argv or "-j" in argv:
        print("ERROR: --jobs was removed; analyses always run serially", file=sys.stderr)
        raise SystemExit(2)

    plots_only = "--plots-only" in argv
    append = "--append" in argv
    reuse_newmark = "--reuse-newmark" in argv
    soe_variants_only = "--soe-variants" in argv
    cudss_dffi_only = "--cudss-dffi-only" in argv
    alpha_close_check_only = "--alpha-close-check-only" in argv
    all_mass_modes = "--all-mass-modes" in argv
    include_incremental = "--no-incremental" not in argv
    engine = _parse_engine(argv)
    explicit_rhos = _parse_rho_args(argv)
    rhos = _resolve_rhos(explicit_rhos)
    numberer = _parse_numberer(argv)

    if reuse_newmark and not append:
        print("ERROR: --reuse-newmark requires --append", file=sys.stderr)
        raise SystemExit(2)

    only_flags = sum([cudss_dffi_only, alpha_close_check_only, soe_variants_only])
    if only_flags > 1 and not (cudss_dffi_only and alpha_close_check_only):
        print(
            "ERROR: use at most one of --cudss-dffi-only, --alpha-close-check-only, --soe-variants "
            "(except --cudss-dffi-only with --alpha-close-check-only for dFFI α-close runs)",
            file=sys.stderr,
        )
        raise SystemExit(2)

    runs = _build_run_list(
        rhos,
        cudss_dffi_only=cudss_dffi_only,
        alpha_close_check_only=alpha_close_check_only,
        soe_variants_only=soe_variants_only,
        include_incremental=include_incremental,
    )
    runs = _filter_reuse_newmark(runs, reuse_newmark=reuse_newmark)

    if all_mass_modes and "--mass-mode" in argv:
        print("ERROR: use either --mass-mode N or --all-mass-modes, not both", file=sys.stderr)
        raise SystemExit(2)

    mass_modes = [0, 1, 2] if all_mass_modes else [_parse_mass_mode(argv)]

    sys.path.insert(0, str(here.parent))
    from plotResults import _parse_figure_names

    figure_names = _parse_figure_names(argv)

    for idx, mass_mode in enumerate(mass_modes):
        print(f"\n=== massMode={mass_mode} ===", flush=True)
        rc = _run_mass_mode_matrix(
            here,
            mass_mode=mass_mode,
            runs=runs,
            engine=engine,
            append=append,
            plots_only=plots_only,
            figure_names=figure_names,
            numberer=numberer,
        )
        if rc != 0:
            raise SystemExit(rc)


if __name__ == "__main__":
    main()
