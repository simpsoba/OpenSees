# -*- coding: utf-8 -*-
"""
Run the Two-Story MRF example for a single integrator.

This script is designed to be invoked by `run_integrators.py` via subprocess so that
if an integrator crashes the interpreter (segfault), the parent process can continue
and still produce comparison plots for the successful runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import two_story_MRF as mrf


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--params", required=True, help='JSON array, e.g. [0.5, 0.25]')
    ap.add_argument("--maxIter", type=int, default=1)
    ap.add_argument("--pFlag", type=int, default=None)
    ap.add_argument("--gm", default=os.path.join(here, "ground_motions", "RSN960_NORTHR_LOS270.AT2"))
    from plot_config import DT_ANALYSIS

    ap.add_argument("--dt_analysis", type=float, default=DT_ANALYSIS)
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--system", default=None, help="Linear SOE override, e.g. CuDSS or FullGeneral")
    ap.add_argument(
        "--cudss-precision",
        default=None,
        help="CuDSS precision mode when using CuDSS (e.g. dFFI for single precision)",
    )
    ap.add_argument(
        "--cudss-ir-n-steps",
        type=int,
        default=0,
        help="CuDSS iterative refinement steps when using dFFI (0 = disabled)",
    )
    ap.add_argument(
        "--test",
        default=None,
        help='Optional JSON convergence test, e.g. {"type":"NormUnbalance","tol":1e-4}',
    )
    ap.add_argument(
        "--mass-mode",
        type=int,
        default=0,
        choices=(0, 1, 2),
        help="0=consistent (-cMass), 1=element lumped, 2=nodal lumped (Cuda adds -diagonalMass)",
    )
    ap.add_argument(
        "--numberer",
        default="RCM",
        choices=("Plain", "RCM", "AMD"),
        help="DOF numberer for gravity and transient analysis (default RCM)",
    )
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
    if args.cudss_precision:
        integrator["cudss_precision"] = args.cudss_precision
    if args.cudss_ir_n_steps > 0:
        integrator["cudss_ir_n_steps"] = args.cudss_ir_n_steps
    integrator["mass_mode"] = args.mass_mode
    integrator["numberer"] = args.numberer
    if args.test:
        try:
            integrator["test"] = json.loads(args.test)
        except Exception as e:
            print(f"ERROR: invalid --test JSON: {e}", file=sys.stderr)
            return 2

    # Create model and damping
    mrf.create_model(
        apply_gravity=True,
        plot_model=False,
        mass_mode=args.mass_mode,
        numberer=args.numberer,
    )
    eigenvalues = mrf.ops.eigen(2)
    omegas = [v**0.5 for v in eigenvalues]
    mrf.setup_rayleigh_damping(omegas[0], omegas[1], 0.02, 0.02, use_NPD=True, plot_rayleigh=False)

    mrf.run_dynamic_analysis(
        gm_file=args.gm,
        dt_analysis=args.dt_analysis,
        scale_factor=args.scale,
        integrator=integrator,
        printA=False,
        dt_gm=None,
    )

    out = mrf._result_folder(integrator)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

