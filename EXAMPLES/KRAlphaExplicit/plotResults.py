#!/usr/bin/env python3
"""Shared plots for KRAlphaExplicit Two-Story MRF (woodbury / modalDampingW layout).

Reference for errors: Newmark (default form D). Each figure uses four rows (integrator trios):
  KR, KR TP, MKR, MKR TP — each row overlays Newmark plus dense/MultiSOE (total form).
Columns are floors (peak-profile columns are disp / vel / accel / drift).
Under each ``figures/rho_*`` folder, ``standard/`` holds total-form runs and
``incrementalAccel/`` holds ``-incrementalAccel`` runs (same filenames in each).
For ``rho_1`` only, ``standardAlphaCloseCheck/`` and ``incrementalAccelAlphaCloseCheck/``
add runs with ``-alphaCloseCheck`` on all explicit integrators (dense and MultiSOE).

Parallelize per-``rho_*`` figure generation: ``python3 plotResults.py --jobs auto``.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # non-GUI backend (required for --jobs thread pool)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

plt.ioff()

FS = 10
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman", "serif"],
        "mathtext.fontset": "cm",
        "font.size": FS,
        "axes.titlesize": FS,
        "axes.labelsize": FS,
        "xtick.labelsize": FS - 1,
        "ytick.labelsize": FS - 1,
        "legend.fontsize": FS - 1,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "lines.scale_dashes": False,
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)

RHOS: List[float] = [0.0, 0.25, 0.50, 0.75, 0.90, 0.99, 0.999, 1.0]
RHO_EQS: List[float] = [0.0, 0.25, 0.50, 0.75, 0.90, 0.99, 0.999, 1.0]
NEWMARK_PARAMS = [0.5, 0.25]
LEGACY_NEWMARK_PARAMS = [0.5, 0.25, "-form", "D"]
EPS = float(np.finfo(float).eps)

# (row label, dense method, MultiSOE method, rho key)
TRIO_GROUPS: List[Tuple[str, str, str, str]] = [
    ("KR", "KRAlphaExplicit", "KRAlphaExplicitMultiSOE", "rho"),
    ("KR TP", "KRAlphaExplicit_TP", "KRAlphaExplicitMultiSOE_TP", "rho"),
    ("MKR", "MKRAlphaExplicit", "MKRAlphaExplicitMultiSOE", "rho_eq"),
    ("MKR TP", "MKRAlphaExplicit_TP", "MKRAlphaExplicitMultiSOE_TP", "rho_eq"),
]

RESP_SYMBOL = {"disp": r"u", "vel": r"\dot{u}", "accel": r"\ddot{u}"}
RESP_UNITS = {"disp": "m", "vel": "m/s", "accel": "m/s²"}
DRIFT_PEAK_LABEL = r"peak $|\Delta u|$ (m)"

COLOR_NEWMARK = "#888888"         # light grey — Newmark GPU (CuDSS)
COLOR_NEWMARK_CPU = "#555555"     # darker grey — Newmark CPU (FullGeneral)
COLOR_EXPLICIT = "#4477AA"        # blue — KRAlphaExplicit (CPU)
COLOR_CUDA_KR = "#EE6677"         # red — CudaKRAlpha
COLOR_CUDA_MKR = "#CCBB44"        # olive — CudaMKRAlpha
COLOR_MULTISOE = "#EE6677"        # red — AlphaExplicitMultiSOE (legacy layout)
COLOR_GRID = "#BBBBBB"

# Minimum log-scale upper limit for |error| and |FFT(error)| panels (shared y across subplots).
ERR_LOG_YMAX_MIN = {"disp": 1.0, "vel": 1.0, "accel": 1.0e7}

# Floor acceleration history y-limits (m/s²); matches two_story_MRF gravity = 9.80665 m/s².
GRAVITY_MS2 = 9.80665
HIST_ACCEL_YLIM_G = 5.0


def result_tag(method: str, params: Sequence) -> str:
    return f"{method}_params-{list(params)}"


def resolve_newmark_tag(results: Path) -> str:
    """Pick Newmark GPU (CuDSS) result folder."""
    for params in (NEWMARK_PARAMS, LEGACY_NEWMARK_PARAMS):
        tag = result_tag("Newmark", params)
        if (results / tag / "disp.out").is_file():
            return tag
    for path in sorted(results.glob("Newmark_params-*")):
        if "FullGeneral" in path.name:
            continue
        if (path / "disp.out").is_file():
            return path.name
    return result_tag("Newmark", NEWMARK_PARAMS)


def resolve_newmark_cpu_tag(results: Path) -> str | None:
    """Pick Newmark CPU (FullGeneral) result folder, if present."""
    for params in (NEWMARK_PARAMS, LEGACY_NEWMARK_PARAMS):
        tag = f"Newmark_FullGeneral_params-{list(params)}"
        p = results / tag / "disp.out"
        if p.is_file() and p.stat().st_size > 0:
            return tag
    for path in sorted(results.glob("Newmark_FullGeneral_params-*")):
        p = path / "disp.out"
        if p.is_file() and p.stat().st_size > 0:
            return path.name
    return None


def _is_incremental_tag(tag: str) -> bool:
    return "incrementalAccel" in tag


def integrator_params(
    rho_val: float, *, incremental: bool = False, alpha_close_check: bool = False
) -> List:
    p: List = [rho_val]
    if incremental:
        p.append("-incrementalAccel")
    if alpha_close_check:
        p.append("-alphaCloseCheck")
    return p


def _rho_param(rho: float, rho_eq: float, kind: str) -> float:
    return rho if kind == "rho" else rho_eq


def trio_tags_total(
    ref: str,
    rho: float,
    rho_eq: float,
    dense_method: str,
    multi_method: str,
    kind: str,
) -> List[str]:
    p = _rho_param(rho, rho_eq, kind)
    return [ref, result_tag(dense_method, [p]), result_tag(multi_method, [p])]


def trio_tags_incr(
    ref: str,
    rho: float,
    rho_eq: float,
    dense_method: str,
    multi_method: str,
    kind: str,
) -> List[str]:
    p = _rho_param(rho, rho_eq, kind)
    pi = integrator_params(p, incremental=True)
    return [ref, result_tag(dense_method, pi), result_tag(multi_method, pi)]


def trio_tags_standard_ac(
    ref: str,
    rho: float,
    rho_eq: float,
    dense_method: str,
    multi_method: str,
    kind: str,
) -> List[str]:
    p = _rho_param(rho, rho_eq, kind)
    pac = integrator_params(p, alpha_close_check=True)
    return [ref, result_tag(dense_method, pac), result_tag(multi_method, pac)]


def trio_tags_incr_ac(
    ref: str,
    rho: float,
    rho_eq: float,
    dense_method: str,
    multi_method: str,
    kind: str,
) -> List[str]:
    p = _rho_param(rho, rho_eq, kind)
    pi = integrator_params(p, incremental=True, alpha_close_check=True)
    return [ref, result_tag(dense_method, pi), result_tag(multi_method, pi)]


def trio_tags(
    ref: str,
    rho: float,
    rho_eq: float,
    dense_method: str,
    multi_method: str,
    kind: str,
    *,
    include_incremental: bool = True,
) -> List[str]:
    tags = trio_tags_total(ref, rho, rho_eq, dense_method, multi_method, kind)
    if include_incremental:
        tags.extend(trio_tags_incr(ref, rho, rho_eq, dense_method, multi_method, kind)[1:])
    return tags


FIG_SUBDIR_STANDARD = "standard"
FIG_SUBDIR_INCREMENTAL = "incrementalAccel"
FIG_SUBDIR_STANDARD_AC = "standardAlphaCloseCheck"
FIG_SUBDIR_INCREMENTAL_AC = "incrementalAccelAlphaCloseCheck"

# (output subfolder under figures/rho_*, tag list builder for one trio row)
PANEL_VARIANTS: List[Tuple[str, object]] = [
    (FIG_SUBDIR_STANDARD, trio_tags_total),
    (FIG_SUBDIR_INCREMENTAL, trio_tags_incr),
]

# rho=1 only: all explicit integrators with -alphaCloseCheck
PANEL_VARIANTS_RHO_ONE_AC: List[Tuple[str, object]] = [
    (FIG_SUBDIR_STANDARD_AC, trio_tags_standard_ac),
    (FIG_SUBDIR_INCREMENTAL_AC, trio_tags_incr_ac),
]


def panel_variants_for_rho(rho: float) -> List[Tuple[str, object]]:
    variants = list(PANEL_VARIANTS)
    if abs(rho - 1.0) < 1e-12:
        variants.extend(PANEL_VARIANTS_RHO_ONE_AC)
    return variants


def resp_time_label(resp: str) -> str:
    return rf"${RESP_SYMBOL[resp]}$"


def resp_peak_label(resp: str) -> str:
    sym = RESP_SYMBOL[resp]
    return rf"peak $|{sym}|$ ({RESP_UNITS[resp]})"


def resp_error_label(resp: str) -> str:
    sym = RESP_SYMBOL[resp]
    return rf"$|{sym}|$ error vs. Newmark"


def resp_fft_error_label(resp: str) -> str:
    sym = RESP_SYMBOL[resp]
    return rf"$|\mathrm{{FFT}}({sym})|$ error vs. Newmark"


def is_diagonal_mass_tag(tag: str) -> bool:
    return "-diagonalMass" in tag or "-lumped" in tag


def legend_label(tag: str) -> str:
    if tag.startswith("Newmark_FullGeneral"):
        return "Newmark (CPU FullGeneral)"
    if tag.startswith("Newmark"):
        return "Newmark (GPU CuDSS)"
    if "MultiSOE" in tag:
        return "AlphaExplicitMultiSOE"
    diag_mass = is_diagonal_mass_tag(tag)
    if tag.startswith("KRAlphaExplicit"):
        return "KRAlphaExplicit (CPU)"
    if tag.startswith("CudaKRAlpha"):
        return "CudaKRAlpha (GPU)" + (" diag. mass" if diag_mass else "")
    if tag.startswith("CudaMKRAlpha"):
        return "CudaMKRAlpha (GPU)" + (" diag. mass" if diag_mass else "")
    return "AlphaExplicit"


def style_axes(ax, *, logy: bool = False) -> None:
    ax.grid(True, which="both" if logy else "major")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_legend(fig, handles, labels, ncol: int = 2) -> None:
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=ncol,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.2,
    )


def legend_proxy_for_tag(tag: str) -> Line2D:
    ls, col, lw, _z = plot_style(tag)
    line = Line2D([0], [0], color=col, lw=lw, label=legend_label(tag))
    apply_line_dashes(line, ls)
    return line


def ordered_cuda_tags(
    ref: str,
    rho: float,
    *,
    ref_cpu: str | None = None,
    include_newmark: bool = True,
    rows: List[Tuple[str, List[str]]] | None = None,
) -> List[str]:
    """Unique integrator tags in panel order; Newmark refs last."""
    if rows is None:
        rows = cuda_panel_rows(ref, rho, ref_cpu=ref_cpu)
    seen: List[str] = []
    for _, row_tags in rows:
        for tag in row_tags:
            if not include_newmark and tag.startswith("Newmark"):
                continue
            if tag not in seen:
                seen.append(tag)
    integrators = [t for t in seen if not t.startswith("Newmark")]
    newmarks = [t for t in seen if t.startswith("Newmark")]
    if include_newmark:
        return integrators + newmarks
    return integrators


def add_figure_legend(fig, tags: Sequence[str], *, nrow: int = 2) -> None:
    """Single figure-level legend above the full subplot grid (KRAlphaSparse layout)."""
    if not tags:
        return
    handles = [legend_proxy_for_tag(t) for t in tags]
    labels = [legend_label(t) for t in tags]
    ncol = (len(handles) + nrow - 1) // nrow
    add_legend(fig, handles, labels, ncol=ncol)


def add_legend_from_ax(fig, ax, *, nrow: int = 2) -> None:
    """Figure legend: integrators first, Newmark last; 2 rows, ncol = ceil(n / 2)."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    first: List[Tuple] = []
    newmark: List[Tuple] = []
    for h, lab in zip(handles, labels):
        if lab.startswith("Newmark"):
            newmark.append((h, lab))
        else:
            first.append((h, lab))
    ordered = first + newmark
    ncol = (len(ordered) + nrow - 1) // nrow
    h_ord, lab_ord = zip(*ordered)
    add_legend(fig, list(h_ord), list(lab_ord), ncol=ncol)


def add_legend_from_grid(fig, axes: np.ndarray, *, nrow: int = 2) -> None:
    """Collect unique legend entries from each row (column 0), integrators then Newmark."""
    seen: set[str] = set()
    handles: List = []
    labels: List[str] = []
    for row in range(axes.shape[0]):
        ax = axes[row, 0]
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab in seen:
                continue
            seen.add(lab)
            handles.append(h)
            labels.append(lab)
    if not handles:
        return
    first: List[Tuple] = []
    newmark: List[Tuple] = []
    for h, lab in zip(handles, labels):
        if lab.startswith("Newmark"):
            newmark.append((h, lab))
        else:
            first.append((h, lab))
    ordered = first + newmark
    ncol = (len(ordered) + nrow - 1) // nrow
    h_ord, lab_ord = zip(*ordered)
    add_legend(fig, list(h_ord), list(lab_ord), ncol=ncol)


def plot_style(tag: str) -> Tuple[str, str, float, int]:
    """Line style, color, width, z-order."""
    if tag.startswith("Newmark_FullGeneral"):
        return ":", COLOR_NEWMARK_CPU, 0.85, 4
    if tag.startswith("Newmark"):
        return "-", COLOR_NEWMARK, 0.85, 3
    if tag.startswith("KRAlphaExplicit"):
        return "-.", COLOR_EXPLICIT, 0.55, 2
    if tag.startswith("CudaMKRAlpha"):
        ls = ":" if is_diagonal_mass_tag(tag) else "--"
        return ls, COLOR_CUDA_MKR, 0.55, 2
    if tag.startswith("CudaKRAlpha"):
        ls = ":" if is_diagonal_mass_tag(tag) else "--"
        return ls, COLOR_CUDA_KR, 0.55, 1
    if "MultiSOE" in tag:
        return "--", COLOR_MULTISOE, 0.55, 1
    return "-.", COLOR_EXPLICIT, 0.55, 2


def apply_line_dashes(line, ls: str) -> None:
    """Dash pattern in points (scale_dashes=False); avoids segment-length artifacts."""
    if ls == "-":
        return
    if ls == "--":
        line.set_dashes([6, 3])
    elif ls == "-.":
        line.set_dashes([5, 2, 1.2, 2])
    elif ls == ":":
        line.set_dashes([2, 2])


def check_uniform_dt(t: np.ndarray, tag: str, dt_expect: float) -> None:
    if t.size < 2:
        return
    dt = np.diff(t)
    if np.max(np.abs(dt - dt_expect)) > 1e-6:
        print(
            f"WARNING: {tag} — non-uniform dt (min={dt.min():.6g}, max={dt.max():.6g}, "
            f"expected {dt_expect:g})",
            flush=True,
        )


def plot_stride_for(dt_analysis: float, dt_plot: float) -> int:
    return max(1, int(round(dt_plot / dt_analysis)))


def _finite_abs_max(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    a = np.asarray(arr, dtype=float).ravel()
    ok = np.isfinite(a)
    if not np.any(ok):
        return 0.0
    return float(np.max(np.abs(a[ok])))


def nrmse_aligned(y_newmark: np.ndarray, y_other: np.ndarray) -> float:
    """NRMSE = mean((Newmark - other)^2) / (max(Newmark) - min(Newmark)) on aligned 1D series."""
    y_newmark = np.asarray(y_newmark, dtype=float).ravel()
    y_other = np.asarray(y_other, dtype=float).ravel()
    n = min(y_newmark.size, y_other.size)
    if n == 0:
        return float("nan")
    y_newmark = y_newmark[:n]
    y_other = y_other[:n]
    diff = y_newmark - y_other
    mse = float(np.mean(diff * diff))
    denom = float(np.max(y_newmark) - np.min(y_newmark))
    if denom < EPS:
        return float("nan")
    return mse / denom


def _format_nrmse(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    if value < 0.01 or value >= 100.0:
        return f"{value:.2e}"
    return f"{value:.3f}"


def annotate_nrmse_history(ax, n_alpha_explicit: float, n_multisoe: float) -> None:
    """Upper-right NRMSE vs Newmark for dense and MultiSOE integrators."""
    label = (
        "NRMSE\n"
        f"AlphaExplicit: {_format_nrmse(n_alpha_explicit)}\n"
        f"AlphaExplicitMultiSOE: {_format_nrmse(n_multisoe)}"
    )
    ax.text(
        0.98,
        0.98,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FS - 2,
        linespacing=1.12,
        color="#333333",
        clip_on=True,
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85),
    )


def _parse_plot_jobs(argv: List[str]) -> int:
    """Parallel worker count for per-rho figure generation; default 1 (sequential)."""
    for i, arg in enumerate(argv):
        if arg in ("--jobs", "-j") and i + 1 < len(argv):
            val = argv[i + 1]
            if val.lower() == "auto":
                return max(1, os.cpu_count() or 1)
            return max(1, int(val))
    return 1


def load_config(example_dir: Path):
    path = example_dir / "plot_config.py"
    spec = importlib.util.spec_from_file_location("plot_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rho_from_result_tag(name: str) -> float | None:
    if "_params-" not in name:
        return None
    tail = name.split("_params-", 1)[1]
    try:
        params = ast.literal_eval(tail)
    except (SyntaxError, ValueError):
        return None
    if isinstance(params, list) and params:
        try:
            return float(params[0])
        except (TypeError, ValueError):
            return None
    return None


def discover_cuda_rhos(results: Path) -> List[float]:
    rhos: set[float] = set()
    if results.is_dir():
        for path in results.iterdir():
            if not path.is_dir():
                continue
            rho = _rho_from_result_tag(path.name)
            if rho is not None:
                rhos.add(rho)
    return sorted(rhos, reverse=True)


def cuda_panel_rows(ref: str, rho: float, *, ref_cpu: str | None = None) -> List[Tuple[str, List[str]]]:
    """Rows for the ops-cuda Two-Story MRF integrator matrix (total form)."""
    return cuda_panel_rows_with_flags(ref, rho, ref_cpu=ref_cpu)


def cuda_panel_rows_with_flags(
    ref: str,
    rho: float,
    *,
    ref_cpu: str | None = None,
    incremental: bool = False,
    alpha_close_check: bool = False,
) -> List[Tuple[str, List[str]]]:
    refs = [ref] + ([ref_cpu] if ref_cpu else [])
    p_cpu = [rho]
    p_cuda = integrator_params(rho, incremental=incremental, alpha_close_check=alpha_close_check)
    return [
        ("KR", [*refs, result_tag("KRAlphaExplicit", p_cpu), result_tag("CudaKRAlpha", p_cuda)]),
        ("MKR", [*refs, result_tag("CudaMKRAlpha", p_cuda)]),
    ]


CUDA_PANEL_VARIANTS: List[Tuple[str, object]] = [
    (FIG_SUBDIR_STANDARD, lambda ref, rho, ref_cpu: cuda_panel_rows_with_flags(ref, rho, ref_cpu=ref_cpu)),
    (
        FIG_SUBDIR_INCREMENTAL,
        lambda ref, rho, ref_cpu: cuda_panel_rows_with_flags(
            ref, rho, ref_cpu=ref_cpu, incremental=True
        ),
    ),
]

CUDA_PANEL_VARIANTS_RHO_ONE_AC: List[Tuple[str, object]] = [
    (
        FIG_SUBDIR_STANDARD_AC,
        lambda ref, rho, ref_cpu: cuda_panel_rows_with_flags(
            ref, rho, ref_cpu=ref_cpu, alpha_close_check=True
        ),
    ),
    (
        FIG_SUBDIR_INCREMENTAL_AC,
        lambda ref, rho, ref_cpu: cuda_panel_rows_with_flags(
            ref, rho, ref_cpu=ref_cpu, incremental=True, alpha_close_check=True
        ),
    ),
]


def cuda_panel_variants_for_rho(rho: float) -> List[Tuple[str, object]]:
    variants = list(CUDA_PANEL_VARIANTS)
    if abs(rho - 1.0) < 1e-12:
        variants.extend(CUDA_PANEL_VARIANTS_RHO_ONE_AC)
    return variants


def _is_cuda_example(results: Path) -> bool:
    return results.is_dir() and any(results.glob("CudaKRAlpha_params-*"))


def load_convergence(results: Path, tag: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (time, iters, final_norm) from results/{tag}/convergence.dat."""
    path = results / tag / "convergence.dat"
    if not path.is_file() or path.stat().st_size == 0:
        return np.array([]), np.array([]), np.array([])
    try:
        data = np.loadtxt(path)
    except (ValueError, OSError):
        return np.array([]), np.array([]), np.array([])
    if data.size == 0:
        return np.array([]), np.array([]), np.array([])
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        return np.array([]), np.array([]), np.array([])
    return data[:, 0], data[:, 1], data[:, 2]


def _plot_convergence_series(
    ax,
    ax_norm,
    tag: str,
    results: Path,
    *,
    plot_stride: int,
) -> bool:
    t, iters, fnorm = load_convergence(results, tag)
    if t.size == 0:
        return False
    if plot_stride > 1:
        t = t[::plot_stride]
        iters = iters[::plot_stride]
        fnorm = fnorm[::plot_stride]
    ls, col, lw, z = plot_style(tag)
    line = ax.plot(t, iters, "-", color=col, lw=lw, zorder=z)[0]
    apply_line_dashes(line, ls)
    norm = np.where(np.isfinite(fnorm), np.maximum(np.abs(fnorm), EPS), np.nan)
    line_n = ax_norm.semilogy(t, norm, "-", color=col, lw=lw, zorder=z)[0]
    apply_line_dashes(line_n, ls)
    return True


def plot_cuda_convergence(
    results: Path,
    figures_root: Path,
    ref: str,
    ref_cpu: str | None,
    *,
    plot_stride: int,
    save_fig,
) -> bool:
    """NR convergence history for Newmark GPU vs CPU only."""
    newmark_tags = [ref] + ([ref_cpu] if ref_cpu else [])
    newmark_tags = [t for t in newmark_tags if load_convergence(results, t)[0].size]
    if not newmark_tags:
        return False

    conv_dir = figures_root / "convergence"
    conv_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.8), sharex=True, layout="constrained", squeeze=False)
    ax_iters, ax_norm = axes[0, 0], axes[1, 0]
    for tag in newmark_tags:
        _plot_convergence_series(ax_iters, ax_norm, tag, results, plot_stride=plot_stride)
    ax_iters.set_ylabel("NR iterations")
    ax_norm.set_ylabel("final unbalance norm")
    style_axes(ax_iters)
    style_axes(ax_norm, logy=True)
    max_iters = 0.0
    for line in ax_iters.get_lines():
        yd = np.asarray(line.get_ydata(), dtype=float)
        yd = yd[np.isfinite(yd)]
        if yd.size:
            max_iters = max(max_iters, float(np.max(yd)))
    if max_iters > 0:
        ax_iters.set_ylim(0.0, max(max_iters * 1.05, 1.0))
        ax_iters.set_yticks(np.arange(0, int(max_iters) + 2))
    add_figure_legend(fig, newmark_tags, nrow=1)
    fig.supxlabel("time (s)")
    save_fig(fig, conv_dir / "newmark.png")
    return True


def run_cuda(example_dir: Path, *, jobs: int = 1) -> int:
    """Plots for Newmark + KRAlphaExplicit + CudaKRAlpha/CudaMKRAlpha."""
    cfg = load_config(example_dir)
    dt_analysis = float(getattr(cfg, "DT_ANALYSIS", 0.005))
    dt_plot = float(getattr(cfg, "DT_PLOT", dt_analysis))
    plot_stride = plot_stride_for(dt_analysis, dt_plot)
    results = example_dir / "results"
    figures_root = example_dir / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)
    ref = resolve_newmark_tag(results)
    ref_cpu = resolve_newmark_cpu_tag(results)
    ref_msg = f"Newmark GPU={ref}"
    if ref_cpu:
        ref_msg += f", CPU={ref_cpu}"
    print(f"Using Newmark references: {ref_msg}", flush=True)

    rhos = discover_cuda_rhos(results)
    if not rhos:
        print("ERROR: no CUDA/KR result folders found under results/", file=sys.stderr)
        return 1

    def load_resp(tag: str, resp: str) -> Tuple[np.ndarray, np.ndarray]:
        path = results / tag / f"{resp}.out"
        if not path.is_file() or path.stat().st_size == 0:
            return np.array([]), np.array([])
        try:
            data = np.loadtxt(path)
        except (ValueError, OSError):
            return np.array([]), np.array([])
        if data.size == 0:
            return np.array([]), np.array([])
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.shape[1] < 2:
            return np.array([]), np.array([])
        return data[:, 0], data[:, 1:]

    def pick_horizontal(y: np.ndarray) -> np.ndarray:
        if y.size == 0:
            return y
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_nodes = len(cfg.RECORD_NODES)
        n_dof = len(cfg.MONITOR_DOFS)
        if y.shape[1] == n_nodes:
            return y
        if y.shape[1] != n_nodes * n_dof:
            return y
        h_ix = cfg.MONITOR_DOFS.index(cfg.HORIZONTAL_DOF)
        cols = [j * n_dof + h_ix for j in range(n_nodes)]
        return y[:, cols]

    def pick_floors(y: np.ndarray) -> np.ndarray:
        y = pick_horizontal(y)
        if y.size == 0 or y.ndim < 2:
            return np.empty((0, 0))
        cix = {node: j for j, node in enumerate(cfg.RECORD_NODES)}
        cols = [cix[n] for n in cfg.PLOT_FLOOR_NODES if n in cix]
        if not cols:
            return np.empty((y.shape[0], 0))
        return y[:, cols]

    def align_with_ref(t_ref, y_ref, t, y, tag):
        if t.size < 2 or t_ref.size < 2:
            n = min(len(t_ref), len(t), y_ref.shape[0], y.shape[0])
            return t_ref[:n], y_ref[:n], t[:n], y[:n]
        t_end = min(float(t_ref[-1]), float(t[-1]))
        mask = t_ref <= t_end + 1e-9
        tr = t_ref[mask]
        ncol = min(y_ref.shape[1], y.shape[1])
        yr = np.column_stack([np.interp(tr, t_ref, y_ref[:, j]) for j in range(ncol)])
        ya = np.column_stack([np.interp(tr, t, y[:, j]) for j in range(ncol)])
        return tr, yr, ya

    def save_fig(fig, path: Path) -> None:
        fig.savefig(path)
        plt.close(fig)
        print(f"Wrote {path}", flush=True)

    any_ok = False
    n_floors = len(cfg.PLOT_FLOOR_NODES)
    floor_labs = list(cfg.FLOOR_LABELS)

    for rho in rhos:
        if load_resp(ref, "disp")[0].size == 0:
            print(f"SKIP rho={rho:g}: missing reference disp.out", flush=True)
            continue

        ref_tags = {ref}
        if ref_cpu:
            ref_tags.add(ref_cpu)
        rho_ylabel = rf"$\rho = {rho:g}$"

        for subdir, row_fn in cuda_panel_variants_for_rho(rho):
            rows = row_fn(ref, rho, ref_cpu)
            out_dir = figures_root / f"rho_{rho:g}" / subdir
            out_dir.mkdir(parents=True, exist_ok=True)
            legend_tags = ordered_cuda_tags(
                ref, rho, ref_cpu=ref_cpu, include_newmark=True, rows=rows
            )
            legend_tags_err = ordered_cuda_tags(
                ref, rho, ref_cpu=ref_cpu, include_newmark=False, rows=rows
            )

            for resp, cmp_fname, err_fname in (
                ("disp", "floor_disp.png", "floor_disp_error.png"),
                ("vel", "floor_vel.png", "floor_vel_error.png"),
                ("accel", "floor_accel.png", "floor_accel_error.png"),
            ):
                t_ref, y_ref = load_resp(ref, resp)
                y_ref = pick_floors(y_ref)
                if t_ref.size == 0 or y_ref.size == 0:
                    print(f"SKIP rho={rho:g}/{subdir}: missing reference {resp}.out", flush=True)
                    continue

                newmark_peak = [_finite_abs_max(y_ref[:, j]) for j in range(y_ref.shape[1])]

                fig, axes = plt.subplots(
                    len(rows),
                    n_floors,
                    figsize=(3.4 * n_floors, 2.5 * len(rows)),
                    sharex=True,
                    layout="constrained",
                    squeeze=False,
                )
                fig_err, axes_err = plt.subplots(
                    len(rows),
                    n_floors,
                    figsize=(3.4 * n_floors, 2.5 * len(rows)),
                    sharex=True,
                    layout="constrained",
                    squeeze=False,
                )

                for row_i, (row_lab, tags) in enumerate(rows):
                    for col_i in range(n_floors):
                        ax = axes[row_i, col_i]
                        ax_err = axes_err[row_i, col_i]
                        for tag in tags:
                            if tag in ref_tags:
                                continue
                            t, y = load_resp(tag, resp)
                            if t.size == 0:
                                continue
                            y = pick_floors(y)
                            if col_i >= y.shape[1]:
                                continue
                            ls, col, lw, z = plot_style(tag)
                            yy = y[:, col_i]
                            if plot_stride > 1:
                                tt = t[::plot_stride]
                                yy = yy[::plot_stride]
                            else:
                                tt = t
                            line = ax.plot(tt, yy, "-", color=col, lw=lw, zorder=z)[0]
                            apply_line_dashes(line, ls)

                            t_r, y_r, y_a = align_with_ref(t_ref, y_ref, t, y, tag)
                            err = np.abs(y_a[:, col_i] - y_r[:, col_i])
                            err = np.where(np.isfinite(err), np.maximum(err, EPS), np.nan)
                            if plot_stride > 1:
                                t_r = t_r[::plot_stride]
                                err = err[::plot_stride]
                            if not np.any(np.isfinite(err)):
                                continue
                            line_e = ax_err.semilogy(
                                t_r, err, "-", color=col, lw=lw, zorder=z
                            )[0]
                            apply_line_dashes(line_e, ls)

                        for ref_tag in ([ref, ref_cpu] if ref_cpu else [ref]):
                            t_nr, y_nr = load_resp(ref_tag, resp)
                            if t_nr.size == 0:
                                continue
                            y_nr = pick_floors(y_nr)
                            if col_i >= y_nr.shape[1]:
                                continue
                            ls, col, lw, z = plot_style(ref_tag)
                            yy_ref = y_nr[:, col_i]
                            tt_ref = t_nr[::plot_stride] if plot_stride > 1 else t_nr
                            yy_ref = yy_ref[::plot_stride] if plot_stride > 1 else yy_ref
                            line = ax.plot(
                                tt_ref, yy_ref, "-", color=col, lw=lw, zorder=z
                            )[0]
                            apply_line_dashes(line, ls)

                        if row_i == 0:
                            ax.set_title(floor_labs[col_i])
                            ax_err.set_title(floor_labs[col_i])
                        if col_i == 0:
                            ax.set_ylabel(row_lab)
                            ax_err.set_ylabel(row_lab)
                        style_axes(ax)
                        if col_i < len(newmark_peak) and newmark_peak[col_i] > 0.0:
                            ylim = 1.5 * newmark_peak[col_i]
                            ax.set_ylim(-ylim, ylim)
                        if ax_err.get_lines():
                            style_axes(ax_err, logy=True)
                            err_hi = 0.0
                            for line in ax_err.get_lines():
                                yd = np.asarray(line.get_ydata(), dtype=float)
                                yd = yd[np.isfinite(yd) & (yd > 0.0)]
                                if yd.size:
                                    err_hi = max(err_hi, float(np.max(yd)))
                            err_floor = ERR_LOG_YMAX_MIN.get(resp, 1.0)
                            ymax = max(err_hi * 1.05, err_floor) if err_hi > 0.0 else err_floor
                            ymax = min(ymax, 1.0e12)
                            ax_err.set_yscale("log")
                            ax_err.set_ylim(EPS, ymax)
                            ax_err.set_autoscaley_on(False)
                        else:
                            style_axes(ax_err)
                            ax_err.set_visible(False)
                        if row_i == len(rows) - 1:
                            ax.set_xlabel("time (s)")
                            ax_err.set_xlabel("time (s)")

                fig.supylabel(f"{resp_time_label(resp)}\n{rho_ylabel}")
                fig_err.supylabel(f"{resp_error_label(resp)}\n{rho_ylabel}")
                add_figure_legend(fig, legend_tags)
                add_figure_legend(fig_err, legend_tags_err, nrow=1)
                fig.supxlabel("time (s)")
                fig_err.supxlabel("time (s)")
                save_fig(fig, out_dir / cmp_fname)
                save_fig(fig_err, out_dir / err_fname)
                any_ok = True

    if plot_cuda_convergence(
        results, figures_root, ref, ref_cpu, plot_stride=plot_stride, save_fig=save_fig
    ):
        any_ok = True

    if not any_ok:
        print("ERROR: no CUDA figures produced — run integrators first.", file=sys.stderr)
        return 1
    return 0


def run(example_dir: Path, *, jobs: int = 1) -> int:
    results = example_dir / "results"
    if _is_cuda_example(results):
        return run_cuda(example_dir, jobs=jobs)
    cfg = load_config(example_dir)
    dt_analysis = float(getattr(cfg, "DT_ANALYSIS", 0.005))
    dt_plot = float(getattr(cfg, "DT_PLOT", dt_analysis))
    plot_stride = plot_stride_for(dt_analysis, dt_plot)
    results = example_dir / "results"
    figures_root = example_dir / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)
    ref = resolve_newmark_tag(results)
    print(f"Using Newmark reference: {ref}", flush=True)

    def recorded_nodes(ncols: int):
        if ncols == len(cfg.RECORD_NODES):
            return list(cfg.RECORD_NODES[:ncols])
        if ncols == len(cfg.RECORD_NODES) - 1 and cfg.RECORD_NODES[0] == 0:
            return list(cfg.RECORD_NODES[1 : ncols + 1])
        return list(cfg.RECORD_NODES[:ncols])

    def col_index_for(ncols: int):
        return {node: j for j, node in enumerate(recorded_nodes(ncols))}

    def pick_horizontal(y: np.ndarray) -> np.ndarray:
        """One column per RECORD_NODES entry (horizontal DOF); pass-through if already reduced."""
        if y.size == 0:
            return y
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_nodes = len(cfg.RECORD_NODES)
        if y.shape[1] == n_nodes:
            return y
        n_dof = len(cfg.MONITOR_DOFS)
        if y.shape[1] != n_nodes * n_dof:
            return y
        h_ix = cfg.MONITOR_DOFS.index(cfg.HORIZONTAL_DOF)
        cols = [j * n_dof + h_ix for j in range(n_nodes)]
        return y[:, cols]

    def pick_floors(y: np.ndarray) -> np.ndarray:
        y = pick_horizontal(y)
        if y.size == 0 or y.ndim < 2:
            return np.empty((0, 0))
        cix = col_index_for(y.shape[1])
        cols = [cix[n] for n in cfg.PLOT_FLOOR_NODES if n in cix]
        if not cols:
            return np.empty((y.shape[0], 0))
        return y[:, cols]

    def floor_labels_for(ncols: int):
        if ncols == len(cfg.PLOT_FLOOR_NODES):
            return list(cfg.FLOOR_LABELS)
        cix = col_index_for(ncols)
        return [lab for node, lab in zip(cfg.PLOT_FLOOR_NODES, cfg.FLOOR_LABELS) if node in cix]

    def profile_y_for_columns(ncols: int) -> np.ndarray:
        nodes = recorded_nodes(ncols)
        if len(cfg.PROFILE_FLOORS) == ncols:
            return np.array(cfg.PROFILE_FLOORS, dtype=float)
        return np.array(nodes, dtype=float)

    def profile_story_bounds(n_stories: int):
        if len(cfg.PROFILE_FLOORS) >= n_stories + 1:
            y_low = np.array(cfg.PROFILE_FLOORS[:n_stories], dtype=float)
            y_hi = np.array(cfg.PROFILE_FLOORS[1 : n_stories + 1], dtype=float)
            return y_low, y_hi
        rn = recorded_nodes(n_stories + 1)
        return np.array(rn[:-1], dtype=float), np.array(rn[1:], dtype=float)

    def plot_peak_drift_stairs(ax, peak_drift, y_low, y_hi, ls, col, lw, zorder, label=None):
        xs, ys = [], []
        for k, d in enumerate(peak_drift):
            y0, y1 = float(y_low[k]), float(y_hi[k])
            if k == 0:
                xs.extend([0.0, d])
                ys.extend([y0, y0])
            else:
                xs.extend([peak_drift[k - 1], d])
                ys.extend([y0, y0])
            xs.extend([d, d])
            ys.extend([y0, y1])
        line = ax.plot(xs, ys, "-", color=col, lw=lw, zorder=zorder, label=label)[0]
        apply_line_dashes(line, ls)

    def save_fig(fig, path: Path) -> None:
        fig.savefig(path)
        plt.close(fig)
        print(f"Wrote {path}", flush=True)

    def load_resp(tag: str, resp: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read {results}/{tag}/{resp}.out (monitor nodes × MONITOR_DOFS per column block)."""
        path = results / tag / f"{resp}.out"
        if not path.is_file() or path.stat().st_size == 0:
            return np.array([]), np.array([])
        try:
            data = np.loadtxt(path)
        except (ValueError, OSError):
            return np.array([]), np.array([])
        if data.size == 0:
            return np.array([]), np.array([])
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.shape[1] < 2:
            return np.array([]), np.array([])
        return data[:, 0], data[:, 1:]

    def align_with_ref(t_ref, y_ref, t, y, tag, *, quiet: bool = False):
        if t.size < 2 or t_ref.size < 2:
            n = min(len(t_ref), len(t), y_ref.shape[0], y.shape[0])
            return t_ref[:n], y_ref[:n], t[:n], y[:n]
        check_uniform_dt(t_ref, ref, dt_analysis)
        check_uniform_dt(t, tag, dt_analysis)
        t_end = min(float(t_ref[-1]), float(t[-1]))
        mask = t_ref <= t_end + 1e-9
        tr = t_ref[mask]
        ncol = min(y_ref.shape[1], y.shape[1])
        yr = np.column_stack([np.interp(tr, t_ref, y_ref[:, j]) for j in range(ncol)])
        ya = np.column_stack([np.interp(tr, t, y[:, j]) for j in range(ncol)])
        if not quiet and len(tr) < len(t_ref):
            print(
                f"WARNING: {tag} vs {ref} — error/FFT truncated at t={t_end:g} s "
                f"({len(tr)} of {len(t_ref)} steps)",
                flush=True,
            )
        return tr, yr, ya

    def tags_rho_lt_one() -> List[str]:
        tags: List[str] = [ref]
        for r, r_eq in zip(RHOS, RHO_EQS):
            if r >= 1.0 - 1e-12:
                continue
            for _, dense_m, multi_m, kind in TRIO_GROUPS:
                tags.extend(trio_tags(ref, r, r_eq, dense_m, multi_m, kind))
        return list(dict.fromkeys(tags))

    def apply_sym_ylim(axes: np.ndarray, ymax: float, *, margin: float = 1.02) -> None:
        if ymax <= 0.0:
            return
        lim = ymax * margin
        for ax in np.atleast_1d(axes).flat:
            ax.set_ylim(-lim, lim)

    def apply_fixed_ylim(axes: np.ndarray, ymin: float, ymax: float) -> None:
        for ax in np.atleast_1d(axes).flat:
            ax.set_ylim(ymin, ymax)

    def apply_log_ylim(
        axes: np.ndarray, ymax: float, *, margin: float = 1.05, ymax_min: float = 0.0
    ) -> None:
        hi = max(ymax, ymax_min)
        if hi <= EPS:
            return
        hi *= margin
        for ax in np.atleast_1d(axes).flat:
            ax.set_ylim(EPS, hi)

    def apply_sym_xlim(ax, xmax: float, *, margin: float = 1.02) -> None:
        if xmax <= 0.0:
            return
        lim = xmax * margin
        ax.set_xlim(-lim, lim)

    def trio_grid(n_floors: int) -> Tuple[plt.Figure, np.ndarray]:
        nrows = len(TRIO_GROUPS)
        ncols = n_floors
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.4 * ncols, 2.5 * nrows),
            sharex=True,
            sharey=True,
            layout="constrained",
            squeeze=False,
        )
        return fig, axes

    def nrmse_vs_newmark(tag: str, resp: str, floor_j: int, t_ref: np.ndarray, y_ref: np.ndarray) -> float:
        """NRMSE of one integrator vs Newmark on the aligned reference time grid."""
        t, y = load_resp(tag, resp)
        if t.size == 0 or y_ref.size == 0:
            return float("nan")
        y = pick_floors(y)
        if floor_j >= y.shape[1] or floor_j >= y_ref.shape[1]:
            return float("nan")
        _, y_newmark, y_other = align_with_ref(t_ref, y_ref, t, y, tag, quiet=True)
        return nrmse_aligned(y_newmark[:, floor_j], y_other[:, floor_j])

    def plot_series_on_ax(
        ax,
        tags: Sequence[str],
        resp: str,
        floor_j: int,
        *,
        t_ref: np.ndarray | None = None,
        logy: bool = False,
        show_legend: bool = False,
    ) -> None:
        plot_fn = ax.semilogy if logy else ax.plot
        for tag in tags:
            t, y = load_resp(tag, resp)
            if t.size == 0:
                continue
            y = pick_floors(y)
            if floor_j >= y.shape[1]:
                continue
            ls, col, lw, z = plot_style(tag)
            yy = y[:, floor_j]
            if t_ref is not None and t.size > 1:
                check_uniform_dt(t, tag, dt_analysis)
                t_end = min(float(t_ref[-1]), float(t[-1]))
                tr = t_ref[t_ref <= t_end + 1e-9]
                yy = np.interp(tr, t, yy)
                t = tr
            if plot_stride > 1:
                t = t[::plot_stride]
                yy = yy[::plot_stride]
            lab = legend_label(tag) if show_legend or logy else None
            if logy:
                line = plot_fn(
                    t, np.maximum(np.abs(yy), EPS), "-", color=col, label=lab, lw=lw, zorder=z
                )[0]
            else:
                line = plot_fn(t, yy, "-", color=col, label=lab, lw=lw, zorder=z)[0]
            apply_line_dashes(line, ls)

    def plot_error_on_ax(ax, ref: str, tag: str, resp: str, floor_j: int, t_ref, y_ref) -> None:
        t, y = load_resp(tag, resp)
        if t.size == 0:
            return
        y = pick_floors(y)
        if floor_j >= y.shape[1]:
            return
        t_r, y_r, y_a = align_with_ref(t_ref, y_ref, t, y, tag)
        ls, col, lw, z = plot_style(tag)
        err = np.maximum(np.abs(y_a[:, floor_j] - y_r[:, floor_j]), EPS)
        if plot_stride > 1:
            t_r = t_r[::plot_stride]
            err = err[::plot_stride]
        line = ax.semilogy(t_r, err, "-", color=col, label=legend_label(tag), lw=lw, zorder=z)[0]
        apply_line_dashes(line, ls)

    def plot_fft_error_on_ax(ax, ref: str, tag: str, resp: str, floor_j: int, t_ref, y_ref) -> None:
        t, y = load_resp(tag, resp)
        if t.size == 0:
            return
        y = pick_floors(y)
        if floor_j >= y.shape[1]:
            return
        t_r, y_r, y_a = align_with_ref(t_ref, y_ref, t, y, tag)
        ls, col, lw, z = plot_style(tag)
        freq, mag = error_fft(t_r, y_a[:, floor_j] - y_r[:, floor_j])
        if freq.size > 1:
            line = ax.semilogy(
                freq[1:], mag[1:] + EPS, "-", color=col, label=legend_label(tag), lw=lw, zorder=z
            )[0]
            apply_line_dashes(line, ls)

    def error_fft(t: np.ndarray, err: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(err)
        dt = dt_analysis if n > 1 else 1.0
        if n > 1:
            dtm = float(np.median(np.diff(t)))
            if abs(dtm - dt_analysis) > 1e-6:
                dt = dtm
        spec = np.fft.rfft(err)
        freq = np.fft.rfftfreq(n, dt)
        mag = np.abs(spec) / n
        if n > 2:
            mag[1:-1] *= 2.0
        return freq, mag

    def compute_global_limits() -> Tuple[dict, dict, dict, dict, float]:
        """|y| max for histories/errors/FFT and profile peaks, over all runs with rho < 1."""
        hist = {r: 0.0 for r in ("disp", "vel", "accel")}
        err = {r: 0.0 for r in ("disp", "vel", "accel")}
        fft = {r: 0.0 for r in ("disp", "vel", "accel")}
        prof = {r: 0.0 for r in ("disp", "vel", "accel")}
        drift_peak = 0.0

        for resp in ("disp", "vel", "accel"):
            t_ref, y_ref = load_resp(ref, resp)
            y_ref_f = pick_floors(y_ref)
            for tag in tags_rho_lt_one():
                t, y = load_resp(tag, resp)
                if y.size == 0:
                    continue
                yf = pick_floors(y)
                hist[resp] = max(hist[resp], _finite_abs_max(yf))
                yh = pick_horizontal(y)
                if yh.size:
                    prof[resp] = max(prof[resp], _finite_abs_max(np.max(np.abs(yh), axis=0)))
                if tag == ref or yf.size == 0 or y_ref_f.size == 0:
                    continue
                t_r, y_r, y_a = align_with_ref(t_ref, y_ref_f, t, yf, tag, quiet=True)
                diff = y_a - y_r
                err[resp] = max(err[resp], _finite_abs_max(diff))
                for j in range(diff.shape[1]):
                    _, mag = error_fft(t_r, diff[:, j])
                    if mag.size > 1:
                        fft[resp] = max(fft[resp], _finite_abs_max(mag[1:]))

        for tag in tags_rho_lt_one():
            _, disp = load_resp(tag, "disp")
            disp = pick_horizontal(disp)
            if disp.size == 0 or disp.shape[1] < 2:
                continue
            drift = disp[:, 1:] - disp[:, :-1]
            drift_peak = max(drift_peak, _finite_abs_max(np.max(np.abs(drift), axis=0)))

        return hist, err, fft, prof, drift_peak

    g_hist, g_err, g_fft, g_prof, g_drift = compute_global_limits()
    limits = (g_hist, g_err, g_fft, g_prof, g_drift)

    def plot_one_rho(rho: float, rho_eq: float) -> bool:
        # Ensure worker threads never pick an interactive GUI backend.
        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg", force=True)
        g_hist_l, g_err_l, g_fft_l, g_prof_l, g_drift_l = limits
        produced = False
        figures = figures_root / f"rho_{rho:g}"
        figures.mkdir(parents=True, exist_ok=True)
        rho_sym = r"\rho_{\mathrm{eq}}" if abs(rho - rho_eq) > 1e-12 else r"\rho"
        rho_ylabel = rf"${rho_sym} = {rho:g}$"

        # --- peak profiles: 4 trio rows × (disp, vel, accel, drift); total + incr copies ---
        profile_cols = (
            ("disp", resp_peak_label("disp")),
            ("vel", resp_peak_label("vel")),
            ("accel", resp_peak_label("accel")),
            ("drift", DRIFT_PEAK_LABEL),
        )
        if load_resp(ref, "disp")[1].size:
            for subdir, tag_fn in panel_variants_for_rho(rho):
                out_dir = figures / subdir
                out_dir.mkdir(parents=True, exist_ok=True)
                fig, axes = plt.subplots(
                    len(TRIO_GROUPS),
                    len(profile_cols),
                    figsize=(6.8, 2.4 * len(TRIO_GROUPS)),
                    layout="constrained",
                    squeeze=False,
                )
                ylim_lo, ylim_hi = None, None
                for row, (trio_lab, dense_m, multi_m, kind) in enumerate(TRIO_GROUPS):
                    tags = tag_fn(ref, rho, rho_eq, dense_m, multi_m, kind)
                    ylab = trio_lab
                    for col, (resp_key, xlab) in enumerate(profile_cols):
                        ax = axes[row, col]
                        if resp_key == "drift":
                            for tag in tags:
                                _, disp = load_resp(tag, "disp")
                                disp = pick_horizontal(disp)
                                if disp.size == 0 or disp.shape[1] < 2:
                                    continue
                                drift = disp[:, 1:] - disp[:, :-1]
                                n = drift.shape[1]
                                y_low, y_hi = profile_story_bounds(n)
                                if ylim_lo is None:
                                    ylim_lo, ylim_hi = float(y_low[0]), float(y_hi[-1])
                                peak_drift = np.max(np.abs(drift), axis=0)
                                ls, col_c, lw, z = plot_style(tag)
                                plot_peak_drift_stairs(
                                    ax,
                                    peak_drift,
                                    y_low,
                                    y_hi,
                                    ls,
                                    col_c,
                                    lw,
                                    z,
                                    label=legend_label(tag) if col == 0 else None,
                                )
                        else:
                            for tag in tags:
                                _, y = load_resp(tag, resp_key)
                                y = pick_horizontal(y)
                                if y.size == 0:
                                    continue
                                floors = profile_y_for_columns(y.shape[1])
                                if ylim_lo is None:
                                    ylim_lo, ylim_hi = floors[0], floors[-1]
                                ls, col_c, lw, z = plot_style(tag)
                                peak = np.max(np.abs(y), axis=0)
                                line = ax.plot(
                                    peak,
                                    floors,
                                    "-",
                                    color=col_c,
                                    label=legend_label(tag) if col == 0 else None,
                                    lw=lw,
                                    zorder=z,
                                )[0]
                                apply_line_dashes(line, ls)
                        ax.axvline(0.0, color=COLOR_GRID, lw=0.8, alpha=0.5, zorder=0)
                        if row == 0:
                            ax.set_title(xlab)
                        if col == 0:
                            ax.set_ylabel(ylab)
                        style_axes(ax)
                        if ylim_lo is not None:
                            ax.set_ylim(ylim_lo, ylim_hi)
                        if resp_key == "drift":
                            apply_sym_xlim(ax, g_drift_l)
                        else:
                            apply_sym_xlim(ax, g_prof_l[resp_key])
                        if row == len(TRIO_GROUPS) - 1:
                            ax.set_xlabel(xlab)
                fig.supylabel(f"{cfg.PROFILE_YLABEL}\n{rho_ylabel}")
                add_legend_from_ax(fig, axes[0, 0])
                save_fig(fig, out_dir / "profile_peak_resp.png")

        panels = (
            ("disp", "floor_disp.png", "floor_disp_error.png", "floor_disp_error_fft.png"),
            ("vel", "floor_vel.png", "floor_vel_error.png", "floor_vel_error_fft.png"),
            ("accel", "floor_accel.png", "floor_accel_error.png", "floor_accel_error_fft.png"),
        )

        for resp, cmp_fname, err_fname, fft_fname in panels:
            if load_resp(ref, resp)[0].size == 0:
                print(f"SKIP rho={rho:g}: missing reference {resp}.out", flush=True)
                continue

            t_ref, y_ref = load_resp(ref, resp)
            y_ref = pick_floors(y_ref)
            check_uniform_dt(t_ref, ref, dt_analysis)
            flabs = floor_labels_for(y_ref.shape[1])
            n_floors = len(flabs)

            for subdir, tag_fn in panel_variants_for_rho(rho):
                out_dir = figures / subdir
                out_dir.mkdir(parents=True, exist_ok=True)
                # History: 4 trio rows × floor columns
                fig, axes = trio_grid(n_floors)
                for row, (trio_lab, dense_m, multi_m, kind) in enumerate(TRIO_GROUPS):
                    tags = tag_fn(ref, rho, rho_eq, dense_m, multi_m, kind)
                    for col in range(n_floors):
                        ax = axes[row, col]
                        plot_series_on_ax(
                            ax,
                            tags,
                            resp,
                            col,
                            t_ref=t_ref,
                            show_legend=(row == 0 and col == 0),
                        )
                        if len(tags) >= 3:
                            annotate_nrmse_history(
                                ax,
                                nrmse_vs_newmark(tags[1], resp, col, t_ref, y_ref),
                                nrmse_vs_newmark(tags[2], resp, col, t_ref, y_ref),
                            )
                        ax.axhline(0.0, color=COLOR_GRID, lw=0.8, alpha=0.5, zorder=0)
                        if row == 0:
                            ax.set_title(flabs[col])
                        if col == 0:
                            ax.set_ylabel(trio_lab)
                        style_axes(ax)
                if resp == "accel":
                    lim = HIST_ACCEL_YLIM_G * GRAVITY_MS2
                    apply_fixed_ylim(axes, -lim, lim)
                else:
                    apply_sym_ylim(axes, g_hist_l[resp])
                fig.supylabel(f"{resp_time_label(resp)}\n{rho_ylabel}")
                add_legend_from_ax(fig, axes[0, 0])
                fig.supxlabel("time (s)")
                save_fig(fig, out_dir / cmp_fname)

                # Error vs Newmark (explicit integrators only; ref excluded)
                fig, axes = trio_grid(n_floors)
                for row, (trio_lab, dense_m, multi_m, kind) in enumerate(TRIO_GROUPS):
                    tags = tag_fn(ref, rho, rho_eq, dense_m, multi_m, kind)[1:]
                    for col in range(n_floors):
                        ax = axes[row, col]
                        for tag in tags:
                            plot_error_on_ax(ax, ref, tag, resp, col, t_ref, y_ref)
                        if row == 0:
                            ax.set_title(flabs[col])
                        if col == 0:
                            ax.set_ylabel(trio_lab)
                        style_axes(ax, logy=True)
                apply_log_ylim(axes, g_err_l[resp], ymax_min=ERR_LOG_YMAX_MIN[resp])
                fig.supylabel(f"{resp_error_label(resp)}\n{rho_ylabel}")
                add_legend_from_ax(fig, axes[0, 0])
                fig.supxlabel("time (s)")
                save_fig(fig, out_dir / err_fname)

                # FFT of error vs Newmark
                fig, axes = trio_grid(n_floors)
                for row, (trio_lab, dense_m, multi_m, kind) in enumerate(TRIO_GROUPS):
                    tags = tag_fn(ref, rho, rho_eq, dense_m, multi_m, kind)[1:]
                    for col in range(n_floors):
                        ax = axes[row, col]
                        for tag in tags:
                            plot_fft_error_on_ax(ax, ref, tag, resp, col, t_ref, y_ref)
                        if row == 0:
                            ax.set_title(flabs[col])
                        if col == 0:
                            ax.set_ylabel(trio_lab)
                        style_axes(ax, logy=True)
                apply_log_ylim(axes, g_fft_l[resp], ymax_min=ERR_LOG_YMAX_MIN[resp])
                fig.supylabel(f"{resp_fft_error_label(resp)}\n{rho_ylabel}")
                add_legend_from_ax(fig, axes[0, 0])
                fig.supxlabel("frequency (Hz)")
                save_fig(fig, out_dir / fft_fname)
            produced = True
        return produced

    rho_pairs = list(zip(RHOS, RHO_EQS))
    any_ok = False
    if jobs <= 1 or len(rho_pairs) <= 1:
        for rho, rho_eq in rho_pairs:
            if plot_one_rho(rho, rho_eq):
                any_ok = True
    else:
        print(f"Plotting {len(rho_pairs)} rho folders with {jobs} workers", flush=True)
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(plot_one_rho, rho, rho_eq): rho for rho, rho_eq in rho_pairs}
            for fut in as_completed(futures):
                rho = futures[fut]
                try:
                    if fut.result():
                        any_ok = True
                except Exception as exc:
                    print(f"WARNING: rho={rho:g} plotting failed: {exc!r}", flush=True)

    if not any_ok:
        print("ERROR: no figures produced — run main integrator sweeps first.", file=sys.stderr)
        return 1
    return 0


def main(argv: List[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    jobs = _parse_plot_jobs(argv)
    return run(Path(__file__).resolve().parent, jobs=jobs)


if __name__ == "__main__":
    sys.exit(main())
