#!/usr/bin/env python3
"""Overlay story displacement time series from sequential, SP, and MP runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NSTORIES = 40
GM_STEPS = 1560
DT = 0.02
GM_DURATION = GM_STEPS * DT
SUBPLOT_HEIGHT_IN = 1.4
FIG_WIDTH_IN = 14
DEFAULT_MODE = "analytical"
PLOT_ROWS = 20
PLOT_COLS = 2


def cases_for_mode(mode: str) -> tuple[tuple[str, str, str], ...]:
    return (
        ("OpenSeesFresco", f"output-{mode}", "-"),
        ("OpenSeesSPFresco", f"output-sp-{mode}", "--"),
        ("OpenSeesMPFresco", f"output-mp-{mode}", ":"),
    )


def find_node_file(out_dir: Path, node: int) -> Path | None:
    path = out_dir / f"node_{node}_disp.out"
    if path.exists():
        return path
    matches = sorted(out_dir.glob(f"node_{node}_disp.out.*"))
    return matches[0] if matches else None


def load_disp(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        return np.array([data[0]]), np.array([data[1]])
    return data[:, 0], data[:, 1]


def node_for_cell(row: int, col: int) -> int:
    return row + 1 + col * PLOT_ROWS


def plot_displacements(
    root: Path,
    *,
    cases: tuple[tuple[str, str, str], ...],
    save_path: Path,
) -> list[str]:
    nrows = PLOT_ROWS
    ncols = PLOT_COLS
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIG_WIDTH_IN, SUBPLOT_HEIGHT_IN * nrows),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )

    legend_handles: list = []
    legend_labels: list[str] = []
    missing: list[str] = []
    t_end = 0.0

    for row in range(nrows):
        for col in range(ncols):
            node = node_for_cell(row, col)
            ax = axes[row, col]
            plotted = False

            for label, folder, linestyle in cases:
                out_dir = root / folder
                node_file = find_node_file(out_dir, node)
                if node_file is None:
                    missing.append(f"{folder}/node_{node}_disp.out")
                    continue
                time, disp = load_disp(node_file)
                if time.size == 0:
                    continue
                t_end = max(t_end, float(time[-1]))
                (line,) = ax.plot(time, disp, linestyle=linestyle, linewidth=1.0, label=label)
                if label not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(label)
                plotted = True

            ax.set_title(f"Node {node}", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axvline(GM_DURATION, color="0.5", linestyle=":", linewidth=0.8, alpha=0.8)
            if not plotted:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="gray",
                )

    for ax in axes.ravel():
        ax.set_xlim(0.0, t_end)

    for col in range(ncols):
        axes[-1, col].set_xlabel("Time (s)", fontsize=9)

    for row in range(nrows):
        axes[row, 0].set_ylabel("Disp", fontsize=9)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="outside upper center",
            ncol=len(legend_labels),
            fontsize=10,
            frameon=False,
        )

    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved {save_path}")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Example directory containing output folders",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="expElementMode label (analytical, local)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Output PNG (default: compare_displacements_<mode>.png)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    save_path = args.save or (root / f"compare_displacements_{args.mode}.png")
    cases = cases_for_mode(args.mode)

    missing = plot_displacements(root, cases=cases, save_path=save_path)

    if missing:
        print(f"Warning: {len(missing)} missing file reference(s), e.g. {missing[0]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
