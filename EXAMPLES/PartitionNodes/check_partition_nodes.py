#!/usr/bin/env python3
"""Validate TestPartitionNodes.tcl MPI stdout for global exactly-once rules."""
from __future__ import annotations

import re
import subprocess
import sys


def run(np: int, exe: str) -> str:
    cmd = ["mpiexec", "-n", str(np), exe, "TestPartitionNodes.tcl"]
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = p.stdout + "\n" + p.stderr
    if p.returncode != 0:
        print(out)
        raise SystemExit(f"OpenSeesMP exited {p.returncode}")
    return out


def sum_key(out: str, case_mode: str, key: str) -> int:
    total = 0
    for ln in out.splitlines():
        m = re.match(rf"FLAG rank=\d+ {re.escape(case_mode)} (.+)$", ln.strip())
        if not m:
            continue
        mm = re.search(rf"{key}=([01])", m.group(1))
        if mm:
            total += int(mm.group(1))
    return total


def sum_mass_owner(out: str, case_mode: str) -> int:
    total = 0
    for ln in out.splitlines():
        m = re.match(
            rf"FLAG rank=\d+ {re.escape(case_mode)} massOwner=([01])$", ln.strip()
        )
        if m:
            total += int(m.group(1))
    return total


def main() -> None:
    exe = sys.argv[1] if len(sys.argv) > 1 else "OpenSeesMP"
    np = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    out = run(np, exe)

    fails = [ln for ln in out.splitlines() if ln.startswith("CHECK FAIL")]
    if fails:
        print("\n".join(fails))
        raise SystemExit(f"{len(fails)} per-rank CHECK FAIL lines")

    errors = []
    for mode in ("custom", "metis"):
        for case, key in (("A", "has99"), ("B", "has98")):
            t = sum_key(out, f"{case}_{mode}", key)
            if t != 1:
                errors.append(
                    f"{case}_{mode}: expected exactly 1 rank with {key}=1, got {t}"
                )

        for case, kmaster, kslave in (
            ("C", "has2", "has97"),
            ("E", "has2", "has95"),
            ("F", "has2", "has94"),
        ):
            masters = sum_key(out, f"{case}_{mode}", kmaster)
            slaves = sum_key(out, f"{case}_{mode}", kslave)
            if masters < 1:
                errors.append(
                    f"{case}_{mode}: master never retained ({kmaster} total {masters})"
                )
            if slaves != masters:
                errors.append(
                    f"{case}_{mode}: slave retention {slaves} != master retention {masters}"
                )

        for case in ("D", "F"):
            owners = sum_mass_owner(out, f"{case}_{mode}")
            if owners != 1:
                errors.append(
                    f"{case}_{mode}: expected exactly 1 massOwner, got {owners}"
                )

        h93 = sum_key(out, f"G_{mode}", "has93")
        h92 = sum_key(out, f"G_{mode}", "has92")
        if h93 != 1 or h92 != 1:
            errors.append(
                f"G_{mode}: expected orphans on exactly one rank (93={h93}, 92={h92})"
            )

    if errors:
        print("GLOBAL CHECK FAIL:")
        print("\n".join(errors))
        print(out)
        raise SystemExit(1)

    print(f"ALL GLOBAL CHECKS PASSED (np={np})")


if __name__ == "__main__":
    main()
