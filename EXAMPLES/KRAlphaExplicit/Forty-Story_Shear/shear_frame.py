# -*- coding: utf-8 -*-
"""Forty-story shear frame — Rayleigh (mass + initial stiffness) + KRAlphaExplicit drivers."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ops_import import ops  # noqa: E402

import model
from plot_config import NODE_TAG_SEED
from shear_frame_lib import setup_rayleigh_T1_and_short_cap, run_dynamic_analysis

NSTORIES = 40
MASS = 1.0
K_BOTTOM = 900.0
K_TOP = 600.0
UY = 0.02
B = 0.01
ZETA = 0.025
NMODES = 3  # modes 0–2 (T1 and T3 for Rayleigh targets)
# Base spring: NPD; upper stories use global Rayleigh (M + Kinit)
NPD_ELEMENTS = (1,)


def create_model() -> list:
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    return model.build(ops, NSTORIES, MASS, K_BOTTOM, K_TOP, UY, B, tag_seed=NODE_TAG_SEED)


def setup_damping() -> None:
    a0, a1, T1, T3, T_short = setup_rayleigh_T1_and_short_cap(
        ops,
        NMODES,
        ZETA,
        npd_elements=NPD_ELEMENTS,
    )
    print(
        f"Rayleigh: zeta={ZETA:.4f} at T={1.5*T1:.4f}s and T={T_short:.4f}s "
        f"(min(0.1*T1={0.1*T1:.4f}s, T3={T3:.4f}s)); a0={a0:.6e}, a1={a1:.6e}"
    )


def run_analysis(
    gm_file: str,
    *,
    gm_dt: float,
    gm_scale: float,
    dt_analysis: float,
    free_vibration_seconds: float,
    integrator: dict,
    monitor_nodes: tuple,
) -> int:
    return run_dynamic_analysis(
        ops,
        gm_file=gm_file,
        gm_dt=gm_dt,
        gm_scale=gm_scale,
        monitor_nodes=monitor_nodes,
        monitor_dof=1,
        dt_analysis=dt_analysis,
        free_vibration_seconds=free_vibration_seconds,
        integrator=integrator,
    )
