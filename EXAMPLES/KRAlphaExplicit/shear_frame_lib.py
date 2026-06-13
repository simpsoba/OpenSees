# -*- coding: utf-8 -*-
"""Shared transient driver for 1-D shear-frame KRAlphaExplicit examples."""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from analysis_utils import write_timing

GRAVITY = 9.80665


def rayleigh_coefficients(
    wi: float, wj: float, zetai: float, zetaj: float
) -> Tuple[float, float]:
    """
    Mass (a0) and initial-stiffness (a1) for ``ops.rayleigh(a0, 0, a1, 0)``.

    Solves the 2×2 modal system (closed form, no linear algebra package)::

        [1/wi   wi] [a0]   [2*zetai]
        [1/wj   wj] [a1] = [2*zetaj]

    so that ζ(wi)=zetai and ζ(wj)=zetaj.
    """
    if wi <= 0.0 or wj <= 0.0:
        raise ValueError("Rayleigh target circular frequencies must be positive")
    if abs(wi - wj) < 1.0e-12 * max(wi, wj):
        raise ValueError("Rayleigh target frequencies wi and wj must differ")

    # det(A) = wj/wi - wi/wj
    det = wj / wi - wi / wj
    a0 = 2.0 * (zetai * wj - zetaj * wi) / det
    a1 = 2.0 * (zetaj / wi - zetai / wj) / det
    return a0, a1


def rayleigh_coefficients_from_eigen(
    eigenvalues: Sequence[float],
    mode_i: int,
    zetai: float,
    mode_j: int,
    zetaj: float,
) -> Tuple[float, float]:
    """Pick modes from ``ops.eigen`` results (λ = ω²) and return (a0, a1)."""
    wi = float(eigenvalues[mode_i]) ** 0.5
    wj = float(eigenvalues[mode_j]) ** 0.5
    return rayleigh_coefficients(wi, wj, zetai, zetaj)


def apply_rayleigh(
    ops,
    a0: float,
    a1: float,
    *,
    npd_elements: Optional[Iterable[int]] = None,
    no_damping_elements: Optional[Iterable[int]] = None,
    zero_k_on_elements: Optional[Iterable[int]] = None,
) -> None:
    """
    ``ops.rayleigh alphaM a0 0.0 betaKinit a1 0.0`` plus per-element factors.

    * ``npd_elements``: like two-story NPD — keep ``a0`` on M, zero K contributions
      (``setElementRayleighDampingFactors(ele, a0, 0, 0, 0)``); other elements keep
      full global Rayleigh unless listed in ``no_damping_elements``.
    * ``no_damping_elements``: optional — zero all Rayleigh on listed elements.
    """
    if zero_k_on_elements is not None and npd_elements is None:
        npd_elements = zero_k_on_elements
    ops.rayleigh(a0, 0.0, a1, 0.0)
    if npd_elements is not None:
        for ele in npd_elements:
            ops.setElementRayleighDampingFactors(ele, a0, 0.0, 0.0, 0.0)
    if no_damping_elements is not None:
        for ele in no_damping_elements:
            ops.setElementRayleighDampingFactors(ele, 0.0, 0.0, 0.0, 0.0)


def setup_rayleigh_at_modes(
    ops,
    nmodes: int,
    mode_i: int,
    zetai: float,
    mode_j: int,
    zetaj: float,
    *,
    npd_elements: Optional[Iterable[int]] = None,
    no_damping_elements: Optional[Iterable[int]] = None,
    zero_k_on_elements: Optional[Iterable[int]] = None,
) -> Tuple[float, float]:
    """Rayleigh from two structural modes (0-based indices into ``ops.eigen``)."""
    w2 = ops.eigen(nmodes)
    a0, a1 = rayleigh_coefficients_from_eigen(w2, mode_i, zetai, mode_j, zetaj)
    apply_rayleigh(
        ops,
        a0,
        a1,
        npd_elements=npd_elements,
        no_damping_elements=no_damping_elements,
        zero_k_on_elements=zero_k_on_elements,
    )
    return a0, a1


def setup_rayleigh_at_periods(
    ops,
    T1: float,
    zeta: float,
    period_factors: Tuple[float, float] = (1.5, 0.1),
    *,
    npd_elements: Optional[Iterable[int]] = None,
    no_damping_elements: Optional[Iterable[int]] = None,
    zero_k_on_elements: Optional[Iterable[int]] = None,
) -> Tuple[float, float]:
    """Rayleigh at equal ζ on periods (f1·T1) and (f2·T1) via the same 2×2 system."""
    f1, f2 = period_factors
    wi = 2.0 * np.pi / (f1 * T1)
    wj = 2.0 * np.pi / (f2 * T1)
    a0, a1 = rayleigh_coefficients(wi, wj, zeta, zeta)
    apply_rayleigh(
        ops,
        a0,
        a1,
        npd_elements=npd_elements,
        no_damping_elements=no_damping_elements,
        zero_k_on_elements=zero_k_on_elements,
    )
    return a0, a1


def _period_from_eigenvalue(lam: float) -> float:
    return 2.0 * np.pi / (float(lam) ** 0.5)


def setup_rayleigh_T1_and_short_cap(
    ops,
    nmodes: int,
    zeta: float,
    *,
    long_period_factor: float = 1.5,
    short_period_factor: float = 0.1,
    third_mode_index: int = 2,
    npd_elements: Optional[Iterable[int]] = None,
    no_damping_elements: Optional[Iterable[int]] = None,
    zero_k_on_elements: Optional[Iterable[int]] = None,
) -> Tuple[float, float, float, float, float]:
    """
    Rayleigh (ζ on both targets) at 1.5·T₁ and at the **shorter** of 0.1·T₁ and T₃.

    T₁ from mode 0; T₃ from ``ops.eigen`` index ``third_mode_index`` (2 = third mode).
    """
    if nmodes < 3:
        raise ValueError("nmodes must be at least 3 (need third mode at index 2)")
    if third_mode_index < 0 or third_mode_index >= nmodes:
        raise ValueError("third_mode_index must satisfy 0 <= index < nmodes")
    w2 = ops.eigen(nmodes)
    if len(w2) <= third_mode_index:
        raise ValueError(f"ops.eigen({nmodes}) returned {len(w2)} values; need mode {third_mode_index}")

    T1 = _period_from_eigenvalue(w2[0])
    T3 = _period_from_eigenvalue(w2[third_mode_index])
    T_short = min(short_period_factor * T1, T3)

    wi = 2.0 * np.pi / (long_period_factor * T1)
    wj = 2.0 * np.pi / T_short
    a0, a1 = rayleigh_coefficients(wi, wj, zeta, zeta)
    apply_rayleigh(
        ops,
        a0,
        a1,
        npd_elements=npd_elements,
        no_damping_elements=no_damping_elements,
        zero_k_on_elements=zero_k_on_elements,
    )
    return a0, a1, T1, T3, T_short


def _is_newmark(integrator_method: str) -> bool:
    return integrator_method == "Newmark"


def _transient_algorithm(integrator_method: str) -> str:
    return "Newton" if _is_newmark(integrator_method) else "Linear"


def _default_pflag(integrator: dict) -> int:
    if _is_newmark(integrator["method"]):
        return 0
    return 5 if integrator["maxIter"] == 1 else 2


def _integrator_ops_test_args(integrator, pFlag):
    max_iter = integrator["maxIter"]
    if "test" in integrator:
        t = integrator["test"]
        if isinstance(t, dict):
            return (
                t["type"],
                float(t["tol"]),
                int(t.get("iter", max_iter)),
                int(t.get("pFlag", pFlag)),
            )
        if isinstance(t, (list, tuple)):
            return (
                t[0],
                float(t[1]),
                int(t[2]) if len(t) > 2 else max_iter,
                int(t[3]) if len(t) > 3 else pFlag,
            )
        raise TypeError("integrator['test'] must be a dict or sequence")
    return ("NormUnbalance", 1.0e-8, max_iter, pFlag)


def _integrator_call_params(integrator: dict) -> list:
    return list(integrator["params"])


def _default_system(integrator_method: str) -> str:
    if integrator_method.startswith("Cuda") or integrator_method == "Newmark":
        return "CuDSS"
    if "MultiSOE" in integrator_method:
        return "CuDSS"
    return "FullGeneral"


def _result_folder(integrator: dict) -> str:
    method = integrator["method"]
    params = integrator["params"]
    system = integrator.get("system")
    if system is not None and not (method == "Newmark" and system == "CuDSS"):
        return f"results/{method}_{system}_params-{params!s}"
    return f"results/{method}_params-{params!s}"


def _set_transient_linear_system(
    integrator_method: str, ops, system: Optional[str] = None
) -> None:
    if system is not None:
        ops.system(system)
        return
    ops.system(_default_system(integrator_method))


def _stage_gm_for_run(gm_source: str, output_folder: str) -> str:
    """Per-run GM copy so parallel Path timeSeries reads do not race."""
    import shutil

    gm_dat = os.path.join(output_folder, "gm.dat")
    shutil.copy2(gm_source, gm_dat)
    return gm_dat


def count_gm_lines(gm_path: str) -> int:
    with open(gm_path, "r") as f:
        return sum(1 for line in f if line.strip())


def run_dynamic_analysis(
    ops,
    *,
    gm_file: str,
    gm_dt: float,
    gm_scale: float,
    monitor_nodes: Sequence[int],
    monitor_dof: int = 1,
    dt_analysis: float,
    free_vibration_seconds: float,
    integrator: dict,
    pattern_tag: int = 2,
    ts_tag: int = 2,
) -> int:
    """Transient analysis with path GM; removes uniform excitation after GM duration."""
    if integrator is None:
        integrator = {"method": "Newmark", "params": [0.5, 0.25], "maxIter": 10}

    n_pts = count_gm_lines(gm_file)
    motion_steps = int(round(n_pts * gm_dt / dt_analysis))
    free_steps = int(round(free_vibration_seconds / dt_analysis))
    n_steps = motion_steps + free_steps

    output_folder = _result_folder(integrator)
    os.makedirs(output_folder, exist_ok=True)
    gm_path = _stage_gm_for_run(gm_file, output_folder)

    ops.wipeAnalysis()
    _set_transient_linear_system(integrator["method"], ops, integrator.get("system"))
    ops.constraints("Plain")
    ops.numberer("Plain")
    pFlag = integrator.get("pFlag", _default_pflag(integrator))
    test_args = _integrator_ops_test_args(integrator, pFlag)
    ops.test(test_args[0], test_args[1], test_args[2], test_args[3])
    ops.algorithm(_transient_algorithm(integrator["method"]))
    ops.integrator(integrator["method"], *_integrator_call_params(integrator))
    ops.analysis("Transient")

    ops.timeSeries(
        "Path", ts_tag, "-filePath", gm_path, "-dt", gm_dt, "-factor", GRAVITY
    )
    ops.pattern(
        "UniformExcitation", pattern_tag, monitor_dof, "-accel", ts_tag, "-fact", gm_scale
    )

    results = open(f"{output_folder}/results.txt", "a+")
    ops.logFile(f"{output_folder}/OpenSees.log", "-noEcho")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results.write(f"{current_time} - Analysis STARTED.\n")
    results.write(
        f"{current_time} - GM {os.path.basename(gm_path)} (dt={gm_dt} s, npts={n_pts}), "
        f"dt_analysis={dt_analysis} s, motion_steps={motion_steps}, free_steps={free_steps}, "
        f"gm_scale={gm_scale}; system {integrator.get('system') or _default_system(integrator['method'])}; "
        f"integrator {integrator['method']} {integrator['params']}.\n"
    )
    results.close()

    nodes = list(monitor_nodes)
    for resp in ("disp", "vel", "accel"):
        ops.recorder(
            "Node",
            "-file",
            f"{output_folder}/{resp}.out",
            "-time",
            "-node",
            *nodes,
            "-dof",
            monitor_dof,
            resp,
        )

    ok = 0
    step = 0
    time_per_step: List[float] = []
    iters_per_step: List[int] = []
    tol_per_step: List[float] = []
    t_wall0 = time.perf_counter()

    def _record_convergence_step() -> None:
        try:
            nit = ops.testIter()
            nit = int(nit[0] if isinstance(nit, (list, tuple)) else nit)
        except Exception:
            nit = 0
        try:
            norms = ops.testNorms()
            norms = list(norms) if isinstance(norms, (list, tuple)) else [float(norms)]
            fnorm = norms[nit - 1] if nit and nit <= len(norms) else (norms[-1] if norms else float("nan"))
        except Exception:
            fnorm = float("nan")
        time_per_step.append(float(ops.getTime()))
        iters_per_step.append(nit)
        tol_per_step.append(fnorm)

    while ok == 0 and step < n_steps:
        if step == motion_steps:
            ops.remove("loadPattern", pattern_tag)
        ok = ops.analyze(1, dt_analysis)
        if ok != 0 and integrator["maxIter"] > 1:
            ops.test(test_args[0], test_args[1], test_args[2] * 100, test_args[3])
            ops.algorithm("ModifiedNewton", "-initial")
            ok = ops.analyze(1, dt_analysis)
            if ok == 0:
                ops.test(test_args[0], test_args[1], test_args[2], test_args[3])
                ops.algorithm(_transient_algorithm(integrator["method"]))
        _record_convergence_step()
        step += 1

    write_timing(output_folder, time.perf_counter() - t_wall0)

    if time_per_step:
        conv = np.column_stack(
            [np.asarray(time_per_step), np.asarray(iters_per_step), np.asarray(tol_per_step)]
        )
        np.savetxt(
            f"{output_folder}/convergence.dat",
            conv,
            header="time iters final_norm",
            comments="# ",
        )

    results = open(f"{output_folder}/results.txt", "a+")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if ok == 0:
        results.write(f"{current_time} - Analysis COMPLETED successfully.\n")
        print("Passed!")
    else:
        results.write(f"{current_time} - Analysis FAILED at t = {ops.getTime()} s.\n")
        print("Failed!")
    results.close()
    ops.remove("recorders")
    print(output_folder)
    return 0 if ok == 0 else 1
