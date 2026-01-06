#!/usr/bin/env python3

"""
Plot timing results from a CSV file using pandas.

Usage
-----
    python plot_timing.py solid_bar_v3.csv timing_plot.png

The CSV file is expected to have at least the following columns:
    solver, num_equations, time_seconds [, status]

Multiple solvers are plotted on the same figure using different axis scalings
in a 2x2 subplot layout:

    linear y / linear x   |   linear y / log x
    log y / linear x      |   log y / log x

Only rows with status == 0 are plotted if a 'status' column is present.
"""

import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


def read_timing_csv(path: str) -> Dict[str, Tuple[List[float], List[float]]]:
    """
    Read timing CSV and return data grouped by solver.

    Returns
    -------
    dict
        {solver_name: ([num_equations], [time_seconds])}
    """
    df = pd.read_csv(path)

    # If 'status' column exists, keep only rows with status == 0
    if "status" in df.columns:
        df = df[df["status"] == 0]

    # Ensure required columns exist
    required_cols = ["solver", "num_equations", "time_seconds"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in CSV: {', '.join(missing)}")

    # Group by solver, sort by num_equations, and convert to lists
    data: Dict[str, Tuple[List[float], List[float]]] = {}
    for solver, g in df.groupby("solver"):
        g_sorted = g.sort_values("num_equations")
        data[solver] = (
            g_sorted["num_equations"].to_list(),
            g_sorted["time_seconds"].to_list(),
        )

    return data


def plot_timing(
    data: Dict[str, Tuple[List[float], List[float]]], output_path: str
) -> None:
    """Create and save a 2x2 grid of timing plots with linear/log axes."""
    if not data:
        raise ValueError("No data to plot.")

    # Separate axes so each subplot can have independent x/y scales
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_lin_lin, ax_lin_logx, ax_logy_lin, ax_logy_logx = axes.ravel()

    # Cycle through matplotlib default colors and markers
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X"]

    for idx, (solver, (xs, ys)) in enumerate(sorted(data.items())):
        if not xs or not ys:
            continue
        marker = markers[idx % len(markers)]
        # Plot on all four subplots; only give labels on the first to avoid duplicate legends
        ax_lin_lin.plot(
            xs,
            ys,
            marker=marker,
            linestyle="-",
            linewidth=1.5,
            markersize=5,
            label=solver,
        )
        ax_lin_logx.plot(
            xs,
            ys,
            marker=marker,
            linestyle="-",
            linewidth=1.5,
            markersize=5,
        )
        ax_logy_lin.plot(
            xs,
            ys,
            marker=marker,
            linestyle="-",
            linewidth=1.5,
            markersize=5,
        )
        ax_logy_logx.plot(
            xs,
            ys,
            marker=marker,
            linestyle="-",
            linewidth=1.5,
            markersize=5,
        )

    # Set axis scales
    ax_lin_lin.set_xscale("linear")
    ax_lin_lin.set_yscale("linear")

    ax_lin_logx.set_xscale("log")
    ax_lin_logx.set_yscale("linear")

    ax_logy_lin.set_xscale("linear")
    ax_logy_lin.set_yscale("log")

    ax_logy_logx.set_xscale("log")
    ax_logy_logx.set_yscale("log")

    # Format x-axis in "k" units (e.g. 50k instead of 50000)
    def _format_k(value: float, _pos: int) -> str:
        if value == 0:
            return "0"
        abs_val = abs(value)
        if abs_val >= 1000:
            val_k = value / 1000.0
            # Keep labels compact; usually around tens of k here
            if abs(val_k) >= 100:
                return f"{val_k:.0f}k"
            elif abs(val_k) >= 10:
                return f"{val_k:.1f}k"
            else:
                return f"{val_k:.2f}k"
        # Below 1000, just show the integer
        return f"{value:.0f}"

    k_formatter = FuncFormatter(_format_k)

    for ax in [ax_lin_lin, ax_lin_logx, ax_logy_lin, ax_logy_logx]:
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

    # Apply "k" formatting only to plots with linear x scaling
    for ax in [ax_lin_lin, ax_logy_lin]:
        ax.xaxis.set_major_formatter(k_formatter)

    # Global labels instead of per-axis titles
    fig.supxlabel("Number of equations")
    fig.supylabel("Time [s]")

    ax_lin_lin.legend(title="Solver")

    fig.tight_layout()

    ext = os.path.splitext(output_path)[1].lower()
    # Use higher DPI for raster formats
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        fig.savefig(output_path, dpi=300)
    else:
        fig.savefig(output_path)

    plt.close(fig)


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        script = os.path.basename(argv[0]) if argv else "plot_timing.py"
        print(f"Usage: python {script} input.csv output.(png|svg)", file=sys.stderr)
        return 1

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve CSV path:
    # - If absolute, use as-is.
    # - If relative and exists in current working directory, use as-is.
    # - Otherwise, look for it relative to the script directory.
    csv_arg = argv[1]
    if os.path.isabs(csv_arg):
        csv_path = csv_arg
    elif os.path.isfile(csv_arg):
        csv_path = csv_arg
    else:
        csv_candidate = os.path.join(script_dir, csv_arg)
        csv_path = csv_candidate

    # Resolve output image path:
    # - If absolute, use as-is.
    # - If relative, place it next to the script.
    out_arg = argv[2]
    if os.path.isabs(out_arg):
        out_path = out_arg
    else:
        out_path = os.path.join(script_dir, out_arg)

    if not os.path.isfile(csv_path):
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    try:
        data = read_timing_csv(csv_path)
        plot_timing(data, out_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Error while processing '{csv_path}': {exc}", file=sys.stderr)
        return 1

    print("-"*80)
    print(f"CSV file:    {csv_path}")
    print(f"Output file: {out_path}")
    print("Status: Done")
    print("-"*80)
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
