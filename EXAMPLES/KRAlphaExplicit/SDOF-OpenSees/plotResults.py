#!/usr/bin/env python3
"""SDOF figures — one panel per method (MRF-style paths), all Δt on each figure vs theory."""

from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SDOF_PANEL_WIDTH_IN = 6.65
SDOF_PANEL_HEIGHT_IN = 2.5

X_LABEL = "time (s)"


def _load_shared_plot_results():
    path = Path(__file__).resolve().parents[1] / "plotResults.py"
    spec = importlib.util.spec_from_file_location("kr_alpha_plot_results", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared plot module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kr_plot = _load_shared_plot_results()
COLOR_NEWMARK = kr_plot.COLOR_NEWMARK
EPS = kr_plot.EPS
ERR_LOG_YMAX_MIN = kr_plot.ERR_LOG_YMAX_MIN
FIG_SUBDIR_INCREMENTAL = kr_plot.FIG_SUBDIR_INCREMENTAL
FIG_SUBDIR_INCREMENTAL_AC = kr_plot.FIG_SUBDIR_INCREMENTAL_AC
FIG_SUBDIR_STANDARD = kr_plot.FIG_SUBDIR_STANDARD
FIG_SUBDIR_STANDARD_AC = kr_plot.FIG_SUBDIR_STANDARD_AC
apply_line_dashes = kr_plot.apply_line_dashes
comparison_slug_for_tag = kr_plot.comparison_slug_for_tag
discover_cuda_rhos = kr_plot.discover_cuda_rhos
integrator_params = kr_plot.integrator_params
plot_stride_for = kr_plot.plot_stride_for
plot_style = kr_plot.plot_style
resp_time_label = kr_plot.resp_time_label
result_tag = kr_plot.result_tag
style_axes = kr_plot.style_axes
_parse_plot_jobs = kr_plot._parse_plot_jobs

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdof  # noqa: E402
from plot_config import DT_CASES, DT_PLOT, DT_THEORY, IC_CASES, T_FINAL  # noqa: E402

DT_MARKERS = {
    0.2: "o",
    0.10: "s",
    0.05: "^",
    0.01: "D",
}

RESP_PANELS = (
    ("disp", "floor_disp"),
    ("vel", "floor_vel"),
    ("accel", "floor_accel"),
)

RowFn = Callable[[float], List[Tuple[str, List[str]]]]


def _dt_marker(dt: float) -> str:
    for key, marker in DT_MARKERS.items():
        if abs(dt - key) < 1e-12:
            return marker
    return "x"


def _markevery(n: int) -> int:
    return max(1, n // 40)


def _plot_series(ax, t, y, tag: str, dt: float, *, plot_stride: int, zorder: int = 2) -> None:
    ls, col, lw, _z = plot_style(tag)
    tt = t[::plot_stride] if plot_stride > 1 else t
    yy = y[::plot_stride] if plot_stride > 1 else y
    me = _markevery(len(tt))
    (line,) = ax.plot(
        tt,
        yy,
        "-",
        color=col,
        lw=lw,
        zorder=zorder,
        marker=_dt_marker(dt),
        ms=2.25,
        markevery=me,
    )
    apply_line_dashes(line, ls)


def _dt_proxy(dt: float) -> Line2D:
    return Line2D(
        [0],
        [0],
        color="#444444",
        lw=0,
        marker=_dt_marker(dt),
        ms=2.75,
        markeredgewidth=0.8,
    )


def _theory_proxy() -> Line2D:
    return Line2D([0], [0], color=COLOR_NEWMARK, lw=0.85, label="Theory")


def _add_dt_legend(fig, *, include_theory: bool) -> None:
    handles: List[Line2D] = []
    labels: List[str] = []
    if include_theory:
        handles.append(_theory_proxy())
        labels.append("Theory")
    handles.extend(_dt_proxy(float(case["dt"])) for case in DT_CASES)
    labels.extend(rf"$\Delta t = {case['dt']:g}\,\mathrm{{s}}$" for case in DT_CASES)
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=min(5, len(handles)),
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, 1.02),
    )


def sdof_panel_rows_with_flags(
    rho: float,
    *,
    incremental: bool = False,
    alpha_close_check: bool = False,
) -> List[Tuple[str, List[str]]]:
    p_cpu = [rho]
    p_cuda = integrator_params(
        rho, incremental=incremental, alpha_close_check=alpha_close_check
    )
    return [
        (
            "KR",
            [
                result_tag("KRAlphaExplicit", p_cpu),
                result_tag("KRAlphaExplicitMultiSOE", p_cuda),
                result_tag("CudaKRAlpha", p_cuda),
            ],
        ),
        (
            "MKR",
            [
                result_tag("MKRAlphaExplicit", p_cpu),
                result_tag("MKRAlphaExplicitMultiSOE", p_cuda),
                result_tag("CudaMKRAlpha", p_cuda),
            ],
        ),
    ]


SDOF_PANEL_VARIANTS: List[Tuple[str, RowFn]] = [
    (FIG_SUBDIR_STANDARD, lambda rho: sdof_panel_rows_with_flags(rho)),
    (
        FIG_SUBDIR_INCREMENTAL,
        lambda rho: sdof_panel_rows_with_flags(rho, incremental=True),
    ),
]

SDOF_PANEL_VARIANTS_RHO_ONE_AC: List[Tuple[str, RowFn]] = [
    (
        FIG_SUBDIR_STANDARD_AC,
        lambda rho: sdof_panel_rows_with_flags(rho, alpha_close_check=True),
    ),
    (
        FIG_SUBDIR_INCREMENTAL_AC,
        lambda rho: sdof_panel_rows_with_flags(
            rho, incremental=True, alpha_close_check=True
        ),
    ),
]


def sdof_panel_variants_for_rho(rho: float) -> List[Tuple[str, RowFn]]:
    variants = list(SDOF_PANEL_VARIANTS)
    if abs(rho - 1.0) < 1e-12:
        variants.extend(SDOF_PANEL_VARIANTS_RHO_ONE_AC)
    return variants


def sdof_row_comparison_specs(row_tags: Sequence[str]) -> List[Tuple[str, str]]:
    """(filename slug, method tag) — one figure per integrator."""
    return [(comparison_slug_for_tag(tag), tag) for tag in row_tags]


def _load_series(results_ic_dt: Path, tag: str, resp: str) -> Tuple[np.ndarray, np.ndarray]:
    path = results_ic_dt / tag / f"{resp}.out"
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
    return data[:, 0], data[:, 1]


def _theory_for_ic(ic_case: dict) -> Tuple[np.ndarray, dict[str, np.ndarray]]:
    u0 = float(ic_case["u0"])
    v0 = float(ic_case["v0"])
    t_th, u_th, v_th, a_th = sdof.theoretical_solution(
        u0, v0, dt=DT_THEORY, t_final=T_FINAL
    )
    return t_th, {"disp": u_th, "vel": v_th, "accel": a_th}


def _discover_dt_tags(results_ic: Path) -> List[str]:
    return [str(case["tag"]) for case in DT_CASES if (results_ic / case["tag"]).is_dir()]


def _discover_rhos(results_ic_dt: Path) -> List[float]:
    return discover_cuda_rhos(results_ic_dt)


def _plot_method_variant(
    *,
    ic_tag: str,
    ic_case: dict,
    rho: float,
    subdir: str,
    row_lab: str,
    slug: str,
    method_tag: str,
    results_root: Path,
    figures_root: Path,
) -> bool:
    results_ic = results_root / ic_tag
    dt_tags = _discover_dt_tags(results_ic)
    if not dt_tags:
        return False

    out_dir = figures_root / ic_tag / f"rho_{rho:g}" / subdir / row_lab
    out_dir.mkdir(parents=True, exist_ok=True)

    t_th0, theory0 = _theory_for_ic(ic_case)
    theory_stride = plot_stride_for(DT_THEORY, DT_PLOT)
    rho_ylabel = rf"$\rho = {rho:g}$"

    produced = False
    for resp, stem in RESP_PANELS:
        y_th_ref = theory0[resp]
        theory_peak = float(np.max(np.abs(y_th_ref))) if y_th_ref.size else 1.0
        ylim = max(1.5 * theory_peak, 1e-12)

        fig, ax = plt.subplots(
            1,
            1,
            figsize=(SDOF_PANEL_WIDTH_IN, SDOF_PANEL_HEIGHT_IN),
            layout="constrained",
        )
        fig_err, ax_err = plt.subplots(
            1,
            1,
            figsize=(SDOF_PANEL_WIDTH_IN, SDOF_PANEL_HEIGHT_IN),
            layout="constrained",
        )

        has_data = False

        tt_th = t_th0[::theory_stride] if theory_stride > 1 else t_th0
        yy_th = theory0[resp][::theory_stride] if theory_stride > 1 else theory0[resp]
        ax.plot(tt_th, yy_th, "-", color=COLOR_NEWMARK, lw=0.85, zorder=3)

        ls, col, lw, z = plot_style(method_tag)

        for dt_case in DT_CASES:
            dt_tag = str(dt_case["tag"])
            if dt_tag not in dt_tags:
                continue
            dt = float(dt_case["dt"])
            stride = plot_stride_for(dt, DT_PLOT)
            results_ic_dt = results_ic / dt_tag
            t, y = _load_series(results_ic_dt, method_tag, resp)
            if t.size == 0:
                continue

            _plot_series(ax, t, y, method_tag, dt, plot_stride=stride)
            has_data = True

            y_i = np.interp(t, t_th0, theory0[resp])
            err = np.abs(y - y_i)
            err = np.where(np.isfinite(err), np.maximum(err, EPS), np.nan)
            if not np.any(np.isfinite(err)):
                continue
            tt_e = t[::stride] if stride > 1 else t
            err = err[::stride] if stride > 1 else err
            me = _markevery(len(tt_e))
            (line_e,) = ax_err.semilogy(
                tt_e,
                err,
                "-",
                color=col,
                lw=lw,
                zorder=z,
                marker=_dt_marker(dt),
                ms=2.25,
                markevery=me,
            )
            apply_line_dashes(line_e, ls)

        if not has_data:
            plt.close(fig)
            plt.close(fig_err)
            continue

        ax.set_xlim(0.0, T_FINAL)
        ax_err.set_xlim(0.0, T_FINAL)
        ax.set_ylim(-ylim, ylim)
        style_axes(ax)
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
            ax_err.set_yscale("log")
            ax_err.set_ylim(EPS, min(ymax, 1.0e12))
            ax_err.set_autoscaley_on(False)
        else:
            style_axes(ax_err)
            ax_err.set_visible(False)

        fig.supylabel(f"{resp_time_label(resp)}\n{rho_ylabel}")
        fig_err.supylabel(rf"$|{kr_plot.RESP_SYMBOL[resp]}|$ error vs theory\n{rho_ylabel}")
        _add_dt_legend(fig, include_theory=True)
        _add_dt_legend(fig_err, include_theory=False)
        fig.supxlabel(X_LABEL)
        fig_err.supxlabel(X_LABEL)

        for fig_obj, fname in ((fig, f"{stem}_{slug}.png"), (fig_err, f"error_{stem}_{slug}.png")):
            path = out_dir / fname
            fig_obj.savefig(path)
            plt.close(fig_obj)
            print(f"Wrote {path}", flush=True)
        produced = True

    return produced


def _plot_rho_variant(
    *,
    ic_tag: str,
    ic_case: dict,
    rho: float,
    subdir: str,
    row_fn: RowFn,
    results_root: Path,
    figures_root: Path,
) -> bool:
    rows = row_fn(rho)
    any_ok = False
    for row_lab, row_tags in rows:
        for slug, method_tag in sdof_row_comparison_specs(row_tags):
            if _plot_method_variant(
                ic_tag=ic_tag,
                ic_case=ic_case,
                rho=rho,
                subdir=subdir,
                row_lab=row_lab,
                slug=slug,
                method_tag=method_tag,
                results_root=results_root,
                figures_root=figures_root,
            ):
                any_ok = True
    return any_ok


def run(example_dir: Path, *, jobs: int = 1) -> int:
    results_root = example_dir / "results"
    figures_root = example_dir / "figures"
    if not results_root.is_dir():
        print("ERROR: no results/ directory — run integrators first.", file=sys.stderr)
        return 1

    tasks: List[tuple] = []
    for ic in IC_CASES:
        ic_tag = ic["tag"]
        results_ic = results_root / ic_tag
        if not results_ic.is_dir():
            print(f"WARNING: missing results/{ic_tag}/ — skip", flush=True)
            continue
        dt_tags = _discover_dt_tags(results_ic)
        if not dt_tags:
            print(f"WARNING: no dt folders under results/{ic_tag}/ — skip", flush=True)
            continue
        ref_dt = results_ic / dt_tags[0]
        rhos = _discover_rhos(ref_dt)
        for rho in rhos:
            for subdir, row_fn in sdof_panel_variants_for_rho(rho):
                tasks.append((ic_tag, ic, rho, subdir, row_fn))

    any_ok = False

    def _work(task):
        ic_tag, ic_case, rho, subdir, row_fn = task
        return _plot_rho_variant(
            ic_tag=ic_tag,
            ic_case=ic_case,
            rho=float(rho),
            subdir=subdir,
            row_fn=row_fn,
            results_root=results_root,
            figures_root=figures_root,
        )

    if jobs <= 1:
        for task in tasks:
            any_ok = _work(task) or any_ok
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = [pool.submit(_work, task) for task in tasks]
            for fut in as_completed(futs):
                any_ok = fut.result() or any_ok

    if not any_ok:
        print("ERROR: no figures produced.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    jobs = _parse_plot_jobs(sys.argv[1:])
    raise SystemExit(run(here, jobs=jobs))
