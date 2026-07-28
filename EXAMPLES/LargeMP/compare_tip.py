#!/usr/bin/env python3
"""Compare LargeMP tip results across serial / METIS (mp) / customPartition.

Typical workflow:
  OpenSees Example.tcl 4 4 20
  mpiexec -n 4 OpenSeesMP Example.tcl 4 4 20              # tip_*.mp + ele_part.np4.map
  mpiexec -n 4 OpenSeesMP ExampleCustomPartition.tcl 4 4 20  # tip_*.custom
  python3 compare_tip.py                    # mp vs serial
  python3 compare_tip.py --kind custom --against mp   # custom vs METIS

Defaults: compare tip_*.mp.rank*.txt against tip_*.serial.txt
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path


def read_vals(path: Path, n: int = 3) -> tuple[float, ...]:
    parts = path.read_text().strip().split()
    if len(parts) < n:
        raise ValueError(f"{path}: expected {n} floats, got {parts!r}")
    return tuple(float(parts[i]) for i in range(n))


def ref_files(directory: Path, prefix: str, against: str) -> list[Path]:
    if against == "serial":
        p = directory / f"{prefix}.serial.txt"
        return [p] if p.is_file() else []
    return sorted(Path(p) for p in glob.glob(str(directory / f"{prefix}.{against}.rank*.txt")))


def test_files(directory: Path, prefix: str, kind: str) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(directory / f"{prefix}.{kind}.rank*.txt")))


def compare_prefix(
    directory: Path,
    prefix: str,
    kind: str,
    against: str,
    tol: float,
    relative: bool,
) -> bool:
    refs = ref_files(directory, prefix, against)
    if not refs:
        print(f"ERROR: missing {against} reference for {prefix}", file=sys.stderr)
        return False
    # For serial, one file; for mp/custom against another parallel kind, use first
    # rank file as reference vector (all ranks should agree on eigen; tip only on owner).
    ref = read_vals(refs[0])
    print(f"{prefix} {against} ({refs[0].name}): " + " ".join(f"{v:.12g}" for v in ref))

    tests = test_files(directory, prefix, kind)
    if not tests:
        print(
            f"ERROR: no {prefix}.{kind}.rank*.txt under {directory}",
            file=sys.stderr,
        )
        return False

    ok = True
    for f in tests:
        vals = read_vals(f)
        diffs = [abs(a - b) for a, b in zip(vals, ref)]
        if relative:
            limits = [tol * max(1.0, abs(r)) for r in ref]
            bad = any(d > lim for d, lim in zip(diffs, limits))
            mode = "rel"
        else:
            bad = max(diffs) > tol
            mode = "abs"
        status = "FAIL" if bad else "OK"
        if bad:
            ok = False
        print(
            f"{f.name}: " + " ".join(f"{v:.12g}" for v in vals) + "  "
            + " ".join(f"|d{i}|={d:.3e}" for i, d in enumerate(diffs))
            + f"  [{status}]"
        )

    label = "eigenvalues" if prefix == "tip_eigen" else "tip"
    if ok:
        print(f"PASS: {prefix} {kind} {label} match {against} within {mode} {tol}")
    else:
        print(
            f"ERROR: {prefix} {kind} {label} mismatch vs {against} ({mode} tol={tol})",
            file=sys.stderr,
        )
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory containing tip_*.txt",
    )
    ap.add_argument(
        "--prefix",
        action="append",
        dest="prefixes",
        help="result prefix (repeatable; default: tip_eigen tip_disp tip_dyn)",
    )
    ap.add_argument(
        "--kind",
        default="mp",
        choices=("mp", "custom"),
        help="which parallel outputs to test (default: mp)",
    )
    ap.add_argument(
        "--against",
        default="serial",
        choices=("serial", "mp", "custom"),
        help="reference set (default: serial)",
    )
    ap.add_argument("--tol", type=float, default=None, help="tolerance")
    args = ap.parse_args()

    if args.kind == args.against:
        print("ERROR: --kind and --against must differ", file=sys.stderr)
        return 1

    prefixes = args.prefixes if args.prefixes else ["tip_eigen", "tip_disp", "tip_dyn"]
    ok = True
    for prefix in prefixes:
        if args.tol is not None:
            tol = args.tol
        else:
            tol = 1e-8 if prefix == "tip_eigen" else 1e-6
        relative = prefix == "tip_eigen"
        if not compare_prefix(args.dir, prefix, args.kind, args.against, tol, relative):
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
