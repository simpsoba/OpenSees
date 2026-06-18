# -*- coding: utf-8 -*-
"""Aggregate timing.txt files from KRAlphaExplicit example result folders into CSV."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

EXAMPLE_DIRS = (
    "Two-Story_MRF",
    "Four-Story_Shear",
    "Forty-Story_Shear",
    "SDOF-OpenSees",
)

_PARAMS_RE = re.compile(r"^(.+)_params-\[(.+)\]$")

# Longest-first so MultiSOE names match before shorter prefixes.
_KNOWN_INTEGRATORS = (
    "MKRAlphaExplicitMultiSOE",
    "KRAlphaExplicitMultiSOE",
    "MKRAlphaExplicit",
    "KRAlphaExplicit",
    "CudaMKRAlpha",
    "CudaKRAlpha",
    "Newmark",
)

_SOE_SUFFIXES = ("CuDSS_dFFI_ir5", "CuDSS_dFFI_ir2", "CuDSS_dFFI", "FullGeneral", "UmfPack", "SuperLU")

_DEFAULT_SOLVER = {
    "Newmark": "CuDSS",
    "KRAlphaExplicit": "FullGeneral",
    "MKRAlphaExplicit": "FullGeneral",
    "KRAlphaExplicitMultiSOE": "CuDSS",
    "MKRAlphaExplicitMultiSOE": "CuDSS",
    "CudaKRAlpha": "CuDSS",
    "CudaMKRAlpha": "CuDSS",
}


def _parse_method_solver(folder_prefix: str) -> Tuple[str, str]:
    """Split result-folder prefix into integrator name and linear SOE solver."""
    for soe in _SOE_SUFFIXES:
        suffix = f"_{soe}"
        if folder_prefix.endswith(suffix):
            base = folder_prefix[: -len(suffix)]
            if base in _KNOWN_INTEGRATORS:
                return base, soe
            return base, soe

    if folder_prefix in _KNOWN_INTEGRATORS:
        return folder_prefix, _DEFAULT_SOLVER.get(folder_prefix, "")

    return folder_prefix, ""


def _parse_result_folder(name: str) -> Tuple[str, str, str, str]:
    """Return (integrator, solver, params, legacy_method) from a results folder name."""
    m = _PARAMS_RE.match(name)
    if not m:
        integrator, solver = _parse_method_solver(name)
        return integrator, solver, "", name
    integrator, solver = _parse_method_solver(m.group(1))
    params = f"[{m.group(2)}]"
    legacy_method = m.group(1)
    return integrator, solver, params, legacy_method


def _read_timing(timing_path: Path) -> Tuple[Optional[float], str]:
    wall: Optional[float] = None
    label = ""
    if not timing_path.is_file():
        return wall, label
    for line in timing_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        key, val = parts[0], parts[1]
        if key == "wall_time_s":
            try:
                wall = float(val)
            except ValueError:
                pass
        elif key == "label":
            label = val
    return wall, label


def collect_rows(
    example_dir: Path,
    *,
    example_name: Optional[str] = None,
    results_subdir: str = "results",
) -> List[dict]:
    example = example_name or example_dir.name
    results_root = example_dir / results_subdir
    if not results_root.is_dir():
        return []

    rows: List[dict] = []
    for folder in sorted(results_root.iterdir()):
        if not folder.is_dir():
            continue
        integrator, solver, params, _legacy = _parse_result_folder(folder.name)
        wall, label = _read_timing(folder / "timing.txt")
        try:
            rel = folder.relative_to(example_dir)
        except ValueError:
            rel = folder
        rows.append(
            {
                "example": example,
                "result_folder": str(rel).replace("\\", "/"),
                "integrator": integrator,
                "solver": solver,
                "params": params,
                "wall_time_s": "" if wall is None else f"{wall:.6f}",
                "label": label,
            }
        )
    return rows


def write_timing_summary(
    example_dir: Path,
    *,
    output_path: Optional[Path] = None,
    example_name: Optional[str] = None,
    results_subdir: str = "results",
) -> Path:
    example_dir = example_dir.resolve()
    out = output_path or (example_dir / "timing_summary.csv")
    rows = collect_rows(example_dir, example_name=example_name, results_subdir=results_subdir)
    fieldnames = [
        "example",
        "result_folder",
        "integrator",
        "solver",
        "params",
        "wall_time_s",
        "label",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}", flush=True)
    return out


def write_all_summaries(
    kralpha_root: Path,
    *,
    examples: Optional[Iterable[str]] = None,
    combined_output: Optional[Path] = None,
) -> Tuple[List[Path], Path]:
    kralpha_root = kralpha_root.resolve()
    names = list(examples) if examples is not None else list(EXAMPLE_DIRS)
    per_example: List[Path] = []
    all_rows: List[dict] = []

    for name in names:
        ex_dir = kralpha_root / name
        if not (ex_dir / "results").is_dir():
            continue
        out = write_timing_summary(ex_dir)
        per_example.append(out)
        all_rows.extend(collect_rows(ex_dir))

    combined = combined_output or (kralpha_root / "timing_summary_all.csv")
    fieldnames = [
        "example",
        "result_folder",
        "integrator",
        "solver",
        "params",
        "wall_time_s",
        "label",
    ]
    with combined.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} combined rows to {combined}", flush=True)
    return per_example, combined


def main(argv: Optional[List[str]] = None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "examples",
        nargs="*",
        help="Example folder names under KRAlphaExplicit (default: all with results/)",
    )
    parser.add_argument(
        "--example-dir",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="Collect a single example by absolute/relative path (repeatable)",
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Skip writing timing_summary_all.csv at KRAlphaExplicit root",
    )
    args = parser.parse_args(argv)

    if args.example_dir:
        for ex in args.example_dir:
            write_timing_summary(ex.resolve())
        return 0

    if args.examples:
        write_all_summaries(here, examples=args.examples, combined_output=None if args.no_combined else here / "timing_summary_all.csv")
        if args.no_combined:
            for name in args.examples:
                ex_dir = here / name
                if (ex_dir / "results").is_dir():
                    write_timing_summary(ex_dir)
        return 0

    _, _ = write_all_summaries(
        here,
        combined_output=None if args.no_combined else here / "timing_summary_all.csv",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
