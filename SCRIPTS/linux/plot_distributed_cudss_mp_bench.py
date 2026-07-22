#!/usr/bin/env python3
"""Plot DistributedCuDSS block3D bench: DOFs vs time, CuDSS/Mumps speedup, histories.

Uses the latest CSVs listed in tests/out/block3d_bench_latest.txt
(or paths passed on the CLI).

  conda run -n py312-gpu python SCRIPTS/linux/plot_distributed_cudss_mp_bench.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "tests" / "out"
LATEST = OUT / "block3d_bench_latest.txt"


def load_paths() -> tuple[Path, Path]:
    if len(sys.argv) >= 3:
        return Path(sys.argv[1]), Path(sys.argv[2])
    if not LATEST.exists():
        raise SystemExit(f"No {LATEST}; pass bench.csv history.csv explicitly")
    lines = [ln.strip() for ln in LATEST.read_text().splitlines() if ln.strip()]
    return Path(lines[0]), Path(lines[1])


def label_for(row: pd.Series) -> str:
    return f"{row['system']} (np={int(row['np'])})"


def plot_timing(bench: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=False)
    for ax, typ, title in zip(
        axes,
        ("static", "transient"),
        ("Static (10 LoadControl steps)", "Transient (0.5 s Newmark)"),
    ):
        sub = bench[bench["type"] == typ].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        for (system, np_), g in sub.groupby(["system", "np"], sort=True):
            g = g.sort_values("ndof")
            ax.plot(
                g["ndof"],
                g["analyze_ms"] / 1000.0,
                marker="o",
                linewidth=1.8,
                label=f"{system} (np={int(np_)})",
            )
        ax.set_xlabel("Number of free DOFs")
        ax.set_ylabel("Analyze wall time (s)")
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls=":", alpha=0.6)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("block3D gravity bench: analyze time vs DOFs", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_cudss_vs_mumps_speedup(bench: pd.DataFrame, out_path: Path) -> None:
    """Speedup of DistributedCuDSS over Mumps at matching np / mesh / analysis type."""
    keys = ["type", "np", "nx", "ny", "nz", "ndof"]
    mumps = (
        bench[bench["system"] == "Mumps"][keys + ["analyze_ms"]]
        .rename(columns={"analyze_ms": "mumps_ms"})
    )
    cudss = (
        bench[bench["system"] == "DistributedCuDSS"][keys + ["analyze_ms"]]
        .rename(columns={"analyze_ms": "cudss_ms"})
    )
    merged = mumps.merge(cudss, on=keys, how="inner")
    if merged.empty:
        print(f"Skip speedup plot (no matching Mumps/DistributedCuDSS rows): {out_path}")
        return

    merged = merged.copy()
    merged["speedup"] = merged["mumps_ms"] / merged["cudss_ms"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for ax, typ, title in zip(
        axes,
        ("static", "transient"),
        ("Static (10 LoadControl steps)", "Transient (0.5 s Newmark)"),
    ):
        sub = merged[merged["type"] == typ].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        for np_, g in sub.groupby("np", sort=True):
            g = g.sort_values("ndof")
            ax.plot(
                g["ndof"],
                g["speedup"],
                marker="o",
                linewidth=1.8,
                label=f"np={int(np_)}",
            )
        ax.axhline(1.0, color="0.45", linestyle="--", linewidth=1.0, label="parity")
        ax.set_xlabel("Number of free DOFs")
        ax.set_ylabel(r"Speedup $t_{\mathrm{Mumps}} / t_{\mathrm{DistributedCuDSS}}$")
        ax.set_title(title)
        ax.set_xscale("log")
        ax.grid(True, which="both", ls=":", alpha=0.6)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("DistributedCuDSS speedup over Mumps (matched np)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_history(hist: pd.DataFrame, out_path: Path, mesh: tuple[int, int, int] | None = None) -> None:
    if mesh is None:
        # Prefer largest mesh present
        meshes = (
            hist[["nx", "ny", "nz", "ndof"]]
            .drop_duplicates()
            .sort_values("ndof")
        )
        row = meshes.iloc[-1]
        mesh = (int(row["nx"]), int(row["ny"]), int(row["nz"]))

    nx, ny, nz = mesh
    sub = hist[(hist["nx"] == nx) & (hist["ny"] == ny) & (hist["nz"] == nz)].copy()
    if sub.empty:
        raise SystemExit(f"No history rows for mesh {nx}x{ny}x{nz}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=False)
    for ax, typ, title in zip(
        axes,
        ("static", "transient"),
        (
            f"Static tip |u| history — mesh {nx}×{ny}×{nz} (10 LoadControl steps, 10×g)",
            f"Transient tip |u| history — mesh {nx}×{ny}×{nz} (0.5 s Newmark about Linear 10×g ramp)",
        ),
    ):
        s = sub[sub["type"] == typ]
        if s.empty:
            ax.set_visible(False)
            continue
        for (system, np_), g in s.groupby(["system", "np"], sort=True):
            g = g.sort_values("step")
            ax.plot(
                g["time"],
                g["tip_norm"],
                marker="o",
                markersize=3.5,
                linewidth=1.6,
                label=f"{system} (np={int(np_)})",
            )
        ax.set_xlabel("Pseudo-time / time (s)")
        ax.set_ylabel(r"Tip $\|u\|$")
        ax.set_title(title)
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend(fontsize=7, loc="best", ncol=2)

        # Quantify agreement vs first series
        refs = []
        for _, g in s.groupby(["system", "np"], sort=True):
            refs.append(g.sort_values("step")["tip_norm"].to_numpy())
        if len(refs) >= 2:
            import numpy as np

            max_spread = max(np.max(np.abs(a - b)) for a in refs for b in refs)
            ax.text(
                0.02,
                0.95,
                f"max |Δ| across solvers = {max_spread:.3e}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, lw=0.5),
            )

    fig.suptitle("Response-history agreement across solvers", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    bench_path, hist_path = load_paths()
    bench = pd.read_csv(bench_path)
    hist = pd.read_csv(hist_path)
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = bench_path.stem.replace("block3d_bench_", "")
    plot_timing(bench, OUT / f"block3d_bench_{stamp}_timing.png")
    plot_cudss_vs_mumps_speedup(bench, OUT / f"block3d_bench_{stamp}_speedup.png")
    plot_history(hist, OUT / f"block3d_bench_{stamp}_history.png")


if __name__ == "__main__":
    main()
