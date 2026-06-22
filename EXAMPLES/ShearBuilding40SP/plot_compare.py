#!/usr/bin/env python3
"""Overlay story displacement time series from sequential, SP, and MP runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NSTORIES = 40
GM_STEPS = 1560
DT = 0.02
GM_DURATION = GM_STEPS * DT
SUBPLOT_HEIGHT_IN = 2.2
FIG_WIDTH_IN = 16
PLOT_ROWS = 20
PLOT_COLS = 2
CASES = (
    ("OpenSeesFresco", "output", "-"),
    ("OpenSeesSPFresco", "output-sp", "--"),
    ("OpenSeesMPFresco", "output-mp", ":"),
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
    t_min: float,
    t_max: float | None,
    save_path: Path,
    title: str,
    mark_gm_end: bool,
) -> list[str]:
    nrows = PLOT_ROWS
    ncols = PLOT_COLS
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIG_WIDTH_IN, SUBPLOT_HEIGHT_IN * nrows),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(title, fontsize=14, y=0.995)

    legend_handles: list = []
    legend_labels: list[str] = []
    missing: list[str] = []
    t_end = t_min

    for row in range(nrows):
        for col in range(ncols):
            node = node_for_cell(row, col)
            ax = axes[row, col]
            plotted = False

            for label, folder, linestyle in CASES:
                out_dir = root / folder
                node_file = find_node_file(out_dir, node)
                if node_file is None:
                    missing.append(f"{folder}/node_{node}_disp.out")
                    continue
                time, disp = load_disp(node_file)
                mask = time >= t_min - 1e-9
                if t_max is not None:
                    mask &= time <= t_max + 1e-9
                time = time[mask]
                disp = disp[mask]
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
            if mark_gm_end and GM_DURATION >= t_min and (t_max is None or GM_DURATION <= t_max):
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

    x_hi = t_max if t_max is not None else t_end
    for ax in axes.ravel():
        ax.set_xlim(t_min, x_hi)

    for col in range(ncols):
        axes[-1, col].set_xlabel("Time (s)", fontsize=9)

    for row in range(nrows):
        axes[row, 0].set_ylabel("Disp", fontsize=9)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(legend_labels),
            bbox_to_anchor=(0.5, 0.98),
            fontsize=10,
            frameon=False,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}  (t = {t_min:.2f} .. {x_hi:.2f} s)")
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
        "--save",
        type=Path,
        default=None,
        help="Full-record PNG (default: <root>/compare_displacements.png)",
    )
    parser.add_argument(
        "--save-free",
        type=Path,
        default=None,
        help="Free-vibration PNG (default: <root>/compare_displacements_freevib.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figures interactively (not implemented; saves only)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    save_full = args.save or (root / "compare_displacements.png")
    save_free = args.save_free or (root / "compare_displacements_freevib.png")

    missing: list[str] = []
    missing += plot_displacements(
        root,
        t_min=0.0,
        t_max=None,
        save_path=save_full,
        title=(
            "Story displacements (nodes 1–40): "
            "OpenSeesFresco vs OpenSeesSPFresco vs OpenSeesMPFresco"
        ),
        mark_gm_end=True,
    )
    missing += plot_displacements(
        root,
        t_min=GM_DURATION,
        t_max=None,
        save_path=save_free,
        title=(
            f"Free vibration (t ≥ {GM_DURATION:.1f} s, nodes 1–40): "
            "OpenSeesFresco vs OpenSeesSPFresco vs OpenSeesMPFresco"
        ),
        mark_gm_end=False,
    )

    if missing:
        print(f"Warning: {len(missing)} missing file reference(s), e.g. {missing[0]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
