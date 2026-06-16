# -*- coding: utf-8 -*-
"""SDOF free vibration — compare explicit integrators against closed-form theory."""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ops_import import ops  # noqa: E402

import numpy as np

from analysis_utils import write_timing

from plot_config import DT_ANALYSIS, FORCE, IC_CASES, MASS, T_FINAL, TN, ZETA


def _is_newmark(method: str) -> bool:
    return method == "Newmark"


def _transient_algorithm(method: str) -> str:
    return "Newton" if _is_newmark(method) else "Linear"


def _default_pflag(integrator: dict) -> int:
    if _is_newmark(integrator["method"]):
        return 0
    return 5 if integrator["maxIter"] == 1 else 2


def _default_system(integrator_method: str) -> str:
    if integrator_method.startswith("Cuda") or integrator_method == "Newmark":
        return "CuDSS"
    if "MultiSOE" in integrator_method:
        return "CuDSS"
    return "FullGeneral"


def _set_transient_linear_system(integrator_method: str, system: Optional[str] = None) -> None:
    if system is not None:
        ops.system(system)
        return
    ops.system(_default_system(integrator_method))


def _result_folder(integrator: dict, ic_tag: str, dt_tag: str) -> str:
    method = integrator["method"]
    params = integrator["params"]
    system = integrator.get("system")
    if system is not None and not (method == "Newmark" and system == "CuDSS"):
        base = f"results/{ic_tag}/{dt_tag}/{method}_{system}_params-{params!s}"
    else:
        base = f"results/{ic_tag}/{dt_tag}/{method}_params-{params!s}"
    return base


def _integrator_call_params(integrator: dict) -> list:
    return list(integrator["params"])


def _stiffness() -> float:
    return MASS * (2.0 * np.pi / TN) ** 2


def theoretical_solution(
    u0: float,
    v0: float,
    *,
    dt: float,
    t_final: float,
    m: float = MASS,
    k: Optional[float] = None,
    zeta: float = ZETA,
    F: float = FORCE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if k is None:
        k = _stiffness()
    ust0 = F / k
    omegan = np.sqrt(k / m)
    omega_d = omegan * np.sqrt(max(0.0, 1.0 - zeta * zeta))
    c = 2.0 * zeta * m * omegan

    t_th = np.arange(0.0, t_final + 0.5 * dt, dt)
    exp_term = np.exp(-c * t_th / (2.0 * m))
    cos_term = np.cos(omega_d * t_th)
    sin_term = np.sin(omega_d * t_th)

    u_th = exp_term * (
        (u0 - ust0) * cos_term + (v0 + c / (2.0 * m) * (u0 - ust0)) / omega_d * sin_term
    ) + ust0

    v_th = (-c / (2.0 * m) * u_th) + exp_term * omega_d * (
        -(u0 - ust0) * sin_term + (v0 + c / (2.0 * m) * (u0 - ust0)) / omega_d * cos_term
    )

    a_th = (-c / m * v_th) - (c / (2.0 * m)) ** 2 * u_th + exp_term * (
        -omega_d**2 * (u0 - ust0) * cos_term
        - omega_d * (v0 + c / (2.0 * m) * (u0 - ust0)) * sin_term
    )
    return t_th, u_th, v_th, a_th


def _setup_node_recorders(output_folder: str) -> None:
    for resp in ("disp", "vel", "accel"):
        ops.recorder(
            "Node",
            "-file",
            f"{output_folder}/{resp}.out",
            "-time",
            "-node",
            2,
            "-dof",
            1,
            resp,
        )


def _record_initial_state() -> int:
    """Static step with zero load increment so recorders capture t=0 ICs."""
    ops.wipeAnalysis()
    ops.system("FullGeneral")
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.test("NormDispIncr", 1.0e-12, 1, 0)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 0.0)
    ops.analysis("Static")
    return ops.analyze(1)


def ic_initial_state(ic_tag: str, dt_analysis: float) -> tuple[float, float]:
    del dt_analysis  # ICs are physical (independent of integrator Δt).
    ic = ic_case_by_tag(ic_tag)
    return float(ic["u0"]), float(ic["v0"])


def run_analysis(
    integrator: dict,
    *,
    ic_tag: str,
    dt_tag: str,
    u0: float,
    v0: float,
    dt_analysis: float = DT_ANALYSIS,
    t_final: float = T_FINAL,
) -> int:
    k = _stiffness()
    beta_k = 0.0
    omega_n = np.sqrt(k / MASS)
    alpha_m = (2.0 * ZETA - beta_k * omega_n) * omega_n
    c = alpha_m * MASS + beta_k * k
    a0 = (FORCE - c * v0 - k * u0) / MASS

    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    ops.node(1, 0.0)
    ops.node(2, 0.0, "-mass", MASS)
    ops.fix(1, 1)
    ops.uniaxialMaterial("Elastic", 1, k)
    ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 1, "-doRayleigh", 1)
    ops.rayleigh(alpha_m, beta_k, 0.0, 0.0)

    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, FORCE)
    ops.setNodeDisp(2, 1, u0, "-commit")
    ops.setNodeVel(2, 1, v0, "-commit")
    ops.setNodeAccel(2, 1, a0, "-commit")

    output_folder = _result_folder(integrator, ic_tag, dt_tag)
    os.makedirs(output_folder, exist_ok=True)
    ops.logFile(f"{output_folder}/OpenSees.log", "-noEcho")

    _setup_node_recorders(output_folder)
    ok = _record_initial_state()
    if ok != 0:
        results = open(f"{output_folder}/results.txt", "a+")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results.write(f"{current_time} - Analysis FAILED recording t=0 static step.\n")
        results.close()
        ops.remove("recorders")
        print("Failed!")
        print(output_folder)
        return 1

    ops.wipeAnalysis()
    _set_transient_linear_system(integrator["method"], integrator.get("system"))
    ops.constraints("Plain")
    ops.numberer("Plain")
    p_flag = integrator.get("pFlag", _default_pflag(integrator))
    max_iter = integrator["maxIter"]
    ops.test("NormDispIncr", 1.0e-12, max_iter, p_flag)
    ops.algorithm(_transient_algorithm(integrator["method"]))
    ops.integrator(integrator["method"], *_integrator_call_params(integrator))
    ops.analysis("Transient")

    n_steps = int(round(t_final / dt_analysis))

    results = open(f"{output_folder}/results.txt", "a+")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results.write(f"{current_time} - Analysis STARTED.\n")
    results.write(
        f"{current_time} - IC {ic_tag}: u0={u0:g}, v0={v0:g}; "
        f"dt={dt_analysis:g} s, n_steps={n_steps}, t_end={n_steps * dt_analysis:g} s; "
        f"system {integrator.get('system') or _default_system(integrator['method'])}; "
        f"integrator {integrator['method']} {integrator['params']}.\n"
    )
    results.close()

    ok = 0
    step = 0
    t_wall0 = time.perf_counter()
    while ok == 0 and step < n_steps:
        ok = ops.analyze(1, dt_analysis)
        if ok != 0 and max_iter > 1:
            ops.test("NormDispIncr", 1.0e-12, max_iter * 100, p_flag)
            ops.algorithm("ModifiedNewton", "-initial")
            ok = ops.analyze(1, dt_analysis)
            if ok == 0:
                ops.test("NormDispIncr", 1.0e-12, max_iter, p_flag)
                ops.algorithm(_transient_algorithm(integrator["method"]))
        step += 1

    write_timing(output_folder, time.perf_counter() - t_wall0)
    ops.remove("recorders")

    results = open(f"{output_folder}/results.txt", "a+")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if ok == 0:
        results.write(f"{current_time} - Analysis COMPLETED successfully.\n")
        print("Passed!")
    else:
        results.write(f"{current_time} - Analysis FAILED at time t = {ops.getTime()} s\n")
        print("Failed!")
    results.close()
    print(output_folder)
    return 0 if ok == 0 else 1


def ic_case_by_tag(tag: str) -> dict:
    for case in IC_CASES:
        if case["tag"] == tag:
            return case
    raise KeyError(f"unknown IC tag: {tag!r}")
