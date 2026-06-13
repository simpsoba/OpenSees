#!/usr/bin/env python3
"""SDOF multipanel figures — same rho/variant layout as Two-Story_MRF, theory reference."""

from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


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
discover_cuda_rhos = kr_plot.discover_cuda_rhos
integrator_params = kr_plot.integrator_params
legend_label = kr_plot.legend_label
ordered_cuda_legend_tags = kr_plot.ordered_cuda_legend_tags
plot_stride_for = kr_plot.plot_stride_for
plot_style = kr_plot.plot_style
resp_error_label = kr_plot.resp_error_label
resp_time_label = kr_plot.resp_time_label
result_tag = kr_plot.result_tag
style_axes = kr_plot.style_axes
_parse_plot_jobs = kr_plot._parse_plot_jobs

FS = kr_plot.FS

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdof  # noqa: E402
from plot_config import DT_CASES, DT_PLOT, DT_THEORY, IC_CASES, T_FINAL  # noqa: E402

X_LABEL = "time (s)"

DT_MARKERS = {
    0.2: "o",
    0.10: "s",
    0.05: "^",
    0.01: "D",
}


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


def _method_proxy(tag: str) -> Line2D:
    ls, col, lw, _z = plot_style(tag)
    line = Line2D([0], [0], color=col, lw=lw)
    apply_line_dashes(line, ls)
    return line


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


def _add_dual_legend(fig, method_tags: List[str], *, include_theory: bool) -> None:
    method_handles: List[Line2D] = []
    method_labels: List[str] = []
    if include_theory:
        method_handles.append(_theory_proxy())
        method_labels.append("Theory")
    for tag in ordered_cuda_legend_tags(method_tags):
        method_handles.append(_method_proxy(tag))
        method_labels.append(legend_label(tag))

    dt_handles = [_dt_proxy(float(case["dt"])) for case in DT_CASES]
    dt_labels = [rf"$\Delta t = {case['dt']:g}\,\mathrm{{s}}$" for case in DT_CASES]

    leg_methods = fig.legend(
        method_handles,
        method_labels,
        loc="outside upper center",
        ncol=min(4, len(method_handles)),
        frameon=False,
        handlelength=2.4,
        columnspacing=1.2,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.add_artist(leg_methods)
    fig.legend(
        dt_handles,
        dt_labels,
        loc="outside upper center",
        ncol=len(dt_handles),
        frameon=False,
        handlelength=1.2,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, 1.01),
    )

RowFn = Callable[[float], List[Tuple[str, List[str]]]]


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


def _theory_at_dt(ic_case: dict, dt: float) -> Tuple[np.ndarray, dict[str, np.ndarray]]:
    u0 = float(ic_case["u0"])
    v0 = float(ic_case["v0"]) if ic_case["v0"] is not None else 1.0 / dt
    t_th, u_th, v_th, a_th = sdof.theoretical_solution(
        u0, v0, dt=DT_THEORY, t_final=T_FINAL
    )
    return t_th, {"disp": u_th, "vel": v_th, "accel": a_th}


def _discover_dt_tags(results_ic: Path) -> List[str]:
    return [str(case["tag"]) for case in DT_CASES if (results_ic / case["tag"]).is_dir()]


def _theory_proxy() -> Line2D:
    return Line2D([0], [0], color=COLOR_NEWMARK, lw=0.85, label="Theory")


def _discover_rhos(results_ic_dt: Path) -> List[float]:
    return discover_cuda_rhos(results_ic_dt)


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
    results_ic = results_root / ic_tag
    rows = row_fn(rho)
    out_dir = figures_root / ic_tag / f"rho_{rho:g}" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    per_dt_theory = ic_case["v0"] is None
    if not per_dt_theory:
        t_th, theory_resp = _theory_at_dt(ic_case, float(DT_CASES[0]["dt"]))
        theory_stride = plot_stride_for(DT_THEORY, DT_PLOT)
    rho_ylabel = rf"$\rho = {rho:g}$"

    legend_tags: List[str] = []
    for _, tags in rows:
        for tag in tags:
            if tag not in legend_tags:
                legend_tags.append(tag)

    dt_tags = _discover_dt_tags(results_ic)
    if not dt_tags:
        return False

    produced = False
    for resp, cmp_fname, err_fname in (
        ("disp", "floor_disp.png", "floor_disp_error.png"),
        ("vel", "floor_vel.png", "floor_vel_error.png"),
        ("accel", "floor_accel.png", "floor_accel_error.png"),
    ):
        y_th_ref = (
            theory_resp[resp]
            if not per_dt_theory
            else _theory_at_dt(ic_case, float(DT_CASES[0]["dt"]))[1][resp]
        )
        theory_peak = float(np.max(np.abs(y_th_ref))) if y_th_ref.size else 1.0
        ylim = max(1.5 * theory_peak, 1e-12)

        fig, axes = plt.subplots(
            len(rows),
            1,
            figsize=(6.65, 2.5 * len(rows)),
            sharex=True,
            layout="constrained",
            squeeze=False,
        )
        fig_err, axes_err = plt.subplots(
            len(rows),
            1,
            figsize=(6.65, 2.5 * len(rows)),
            sharex=True,
            layout="constrained",
            squeeze=False,
        )

        for row_i, (row_lab, tags) in enumerate(rows):
            ax = axes[row_i, 0]
            ax_err = axes_err[row_i, 0]

            if not per_dt_theory:
                tt_th = t_th[::theory_stride] if theory_stride > 1 else t_th
                yy_th = theory_resp[resp][::theory_stride] if theory_stride > 1 else theory_resp[resp]
                ax.plot(tt_th, yy_th, "-", color=COLOR_NEWMARK, lw=0.85, zorder=3)

            for dt_case in DT_CASES:
                dt_tag = str(dt_case["tag"])
                if dt_tag not in dt_tags:
                    continue
                dt = float(dt_case["dt"])
                stride = plot_stride_for(dt, DT_PLOT)
                results_ic_dt = results_ic / dt_tag
                if per_dt_theory:
                    t_th_d, theory_d = _theory_at_dt(ic_case, dt)
                    th_stride = plot_stride_for(DT_THEORY, DT_PLOT)
                    tt_d = t_th_d[::th_stride] if th_stride > 1 else t_th_d
                    yy_d = theory_d[resp][::th_stride] if th_stride > 1 else theory_d[resp]
                    ax.plot(tt_d, yy_d, "-", color=COLOR_NEWMARK, lw=0.65, alpha=0.55, zorder=2)
                for tag in tags:
                    t, y = _load_series(results_ic_dt, tag, resp)
                    if t.size == 0:
                        continue
                    _plot_series(ax, t, y, tag, dt, plot_stride=stride)

                    if per_dt_theory:
                        t_th_use, theory_use = _theory_at_dt(ic_case, dt)
                        y_th_line = theory_use[resp]
                    else:
                        t_th_use, y_th_line = t_th, theory_resp[resp]
                    y_i = np.interp(t, t_th_use, y_th_line)
                    err = np.abs(y - y_i)
                    err = np.where(np.isfinite(err), np.maximum(err, EPS), np.nan)
                    if not np.any(np.isfinite(err)):
                        continue
                    ls, col, lw, z = plot_style(tag)
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

            ax.set_ylabel(row_lab)
            ax_err.set_ylabel(row_lab)
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
        fig_err.supylabel(f"{resp_error_label(resp)} vs theory\n{rho_ylabel}")
        _add_dual_legend(fig, legend_tags, include_theory=True)
        _add_dual_legend(fig_err, legend_tags, include_theory=False)
        fig.supxlabel(X_LABEL)
        fig_err.supxlabel(X_LABEL)

        for fig_obj, fname in ((fig, cmp_fname), (fig_err, err_fname)):
            path = out_dir / fname
            fig_obj.savefig(path)
            plt.close(fig_obj)
            print(f"Wrote {path}", flush=True)
        produced = True

    return produced


def run(example_dir: Path, *, jobs: int = 1) -> int:
    results_root = example_dir / "results"
    figures_root = example_dir / "figures"
    if not results_root.is_dir():
        print("ERROR: no results/ directory — run integrators first.", file=sys.stderr)
        return 1

    any_ok = False

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

    def _work(task):
        ic_tag, ic_case, rho, subdir, row_fn = task
        return _plot_rho_variant(
            ic_tag=ic_tag,
            ic_case=ic_case,
            rho=float(rho),
            subdir=str(subdir),
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
