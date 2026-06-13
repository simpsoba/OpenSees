#!/usr/bin/env python3
"""Run one integrator on a shear frame example (subprocess worker)."""

from __future__ import annotations

import argparse
import json
import os
import sys

import shear_frame as sf
from plot_config import DT_ANALYSIS, FREE_VIBRATION_SECONDS, GM_DT, GM_SCALE
from shear_frame_lib import count_gm_lines


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--maxIter", type=int, default=1)
    ap.add_argument("--pFlag", type=int, default=None)
    ap.add_argument("--system", default=None)
    ap.add_argument("--gm", default=os.path.join(here, "tabasFN.txt"))
    ap.add_argument("--dt_analysis", type=float, default=DT_ANALYSIS)
    ap.add_argument("--scale", type=float, default=GM_SCALE)
    ap.add_argument("--gm_dt", type=float, default=GM_DT)
    args = ap.parse_args()

    try:
        params = json.loads(args.params)
        if not isinstance(params, list):
            raise ValueError("params must be a JSON list")
    except Exception as e:
        print(f"ERROR: invalid --params: {e}", file=sys.stderr)
        return 2

    integrator = {"method": args.method, "params": params, "maxIter": args.maxIter}
    if args.pFlag is not None:
        integrator["pFlag"] = args.pFlag
    if args.system:
        integrator["system"] = args.system

    free_sec = FREE_VIBRATION_SECONDS
    if free_sec <= 0.0:
        free_sec = count_gm_lines(args.gm) * args.gm_dt

    monitor_nodes = tuple(sf.create_model())
    sf.setup_damping()
    return sf.run_analysis(
        args.gm,
        gm_dt=args.gm_dt,
        gm_scale=args.scale,
        dt_analysis=args.dt_analysis,
        free_vibration_seconds=free_sec,
        integrator=integrator,
        monitor_nodes=monitor_nodes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
