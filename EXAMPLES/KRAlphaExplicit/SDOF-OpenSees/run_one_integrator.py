# -*- coding: utf-8 -*-
"""Run one SDOF case (single integrator + IC + Δt) via subprocess."""

from __future__ import annotations

import argparse
import json
import os
import sys

import sdof


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--params", required=True, help='JSON array, e.g. [0.5]')
    ap.add_argument("--maxIter", type=int, default=1)
    ap.add_argument("--pFlag", type=int, default=None)
    ap.add_argument("--system", default=None, help="Linear SOE override, e.g. CuDSS or FullGeneral")
    ap.add_argument("--ic", required=True, help="IC tag, e.g. init_disp or init_vel")
    ap.add_argument("--dt_tag", required=True, help="Results subfolder tag, e.g. dt_0.2")
    from plot_config import DT_ANALYSIS

    ap.add_argument("--dt_analysis", type=float, default=DT_ANALYSIS)
    args = ap.parse_args()

    try:
        params = json.loads(args.params)
        if not isinstance(params, list):
            raise ValueError("params must be a JSON list")
    except Exception as e:
        print(f"ERROR: invalid --params: {e}", file=sys.stderr)
        return 2

    try:
        u0, v0 = sdof.ic_initial_state(args.ic, args.dt_analysis)
    except KeyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    integrator = {"method": args.method, "params": params, "maxIter": args.maxIter}
    if args.pFlag is not None:
        integrator["pFlag"] = args.pFlag
    if args.system:
        integrator["system"] = args.system

    return sdof.run_analysis(
        integrator,
        ic_tag=args.ic,
        dt_tag=args.dt_tag,
        u0=u0,
        v0=v0,
        dt_analysis=args.dt_analysis,
    )


if __name__ == "__main__":
    raise SystemExit(main())
