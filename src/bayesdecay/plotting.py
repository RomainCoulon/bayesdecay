"""Figures: fitted activity curve, convergence trace, and posterior marginals."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

COLOR_BLUE = "#2a78d6"     # MAP estimate / posterior
COLOR_ORANGE = "#eb6834"   # estimated background
COLOR_AQUA = "#1baf7a"     # log-linear reference fit
COLOR_TRUE = "#52514e"     # true value (validation runs only)
COLOR_GRID = "#e1e0d9"
COLOR_TICK = "#898781"
COLOR_TEXT = "#0b0b0b"

_STYLE = {
    "font.family": "sans-serif",
    "axes.edgecolor": COLOR_GRID,
    "axes.labelcolor": COLOR_TEXT,
    "text.color": COLOR_TEXT,
    "xtick.color": COLOR_TICK,
    "ytick.color": COLOR_TICK,
    "axes.grid": True,
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _activity_uncertainty(result, t_plot):
    """GUM law of propagation of uncertainty applied to cov(A0, half_life), for
    a(t) = A0 * exp(-lambda * t)."""
    lam = result.decay_constant
    a_lin = result.A0 * np.exp(-lam * t_plot)
    c_A0 = np.exp(-lam * t_plot)
    c_t12 = a_lin * np.log(2) * t_plot / result.half_life**2
    var = c_A0**2 * result.cov[0, 0] + c_t12**2 * result.cov[1, 1] + 2 * c_A0 * c_t12 * result.cov[0, 1]
    return np.sqrt(np.maximum(var, 0.0))


def plot_activity(result, t_max, bin_edges, counts, true_values=None):
    """Counting-rate curve: fitted (MAP) vs. log-linear reference, plus the binned
    counts histogram. If ``true_values`` (dict with A0/half_life/background) is given,
    the true activity curve is overlaid -- for validation runs against simulated data."""
    with plt.rc_context(_STYLE):
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )

        t_plot = np.linspace(0, t_max, 500)
        activity_fit = result.A0 * np.exp(-result.decay_constant * t_plot)
        activity_loglinear = result.A0_loglinear * np.exp(-(np.log(2) / result.half_life_loglinear) * t_plot)
        u_activity = _activity_uncertainty(result, t_plot)

        if true_values is not None:
            activity_true = true_values["A0"] * np.exp(-np.log(2) / true_values["half_life"] * t_plot)
            ax_top.plot(t_plot, activity_true, color=COLOR_TRUE, linestyle="--", linewidth=2, label="True activity")

        ax_top.fill_between(
            t_plot, activity_fit - u_activity, activity_fit + u_activity,
            color=COLOR_BLUE, alpha=0.18, linewidth=0, label="Propagated uncertainty (±1σ)",
        )
        ax_top.plot(t_plot, activity_fit, color=COLOR_BLUE, linewidth=2, label="Fitted activity (Bayesian MAP)")
        ax_top.plot(
            t_plot, activity_loglinear, color=COLOR_AQUA, linestyle="-.", linewidth=1.5,
            label=f"Log-linear regression (A0={result.A0_loglinear:.0f}, T1/2={result.half_life_loglinear:.1f} s)",
        )
        ax_top.axhline(
            result.background, color=COLOR_ORANGE, linestyle=":", linewidth=1.5,
            label=f"Estimated background B = {result.background:.1f} ± {result.u_background:.1f} cps",
        )
        ax_top.set_ylabel("Count rate (cps)")
        ax_top.set_title("Radioactive decay: Bayesian estimation of A0, T1/2 and B")
        ax_top.legend(frameon=False)

        bin_width = bin_edges[1:] - bin_edges[:-1]
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax_bottom.bar(bin_centers, counts, width=bin_width, color=COLOR_BLUE, alpha=0.6, align="center")
        ax_bottom.set_xlabel("Time (s)")
        ax_bottom.set_ylabel("Counts / channel")

        fig.tight_layout()
    return fig


def plot_convergence(result, true_values=None):
    """A0, T1/2, B re-estimated using only the first k channels, for growing k."""
    cutoffs = result.convergence_cutoff()
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

        series = [
            (result.trace_A0, "A0 (cps)", true_values.get("A0") if true_values else None, cutoffs["A0"]),
            (result.trace_half_life, "T1/2 (s)", true_values.get("half_life") if true_values else None, cutoffs["half_life"]),
            (result.trace_background, "B (cps)", true_values.get("background") if true_values else None, cutoffs["background"]),
        ]
        for ax, (trace, ylabel, true_val, cutoff) in zip(axes, series):
            if cutoff is not None and cutoff > result.trace_t[0]:
                ax.axvspan(result.trace_t[0], cutoff, color=COLOR_TICK, alpha=0.12, linewidth=0)
            ax.plot(result.trace_t, trace, color=COLOR_BLUE, linewidth=2)
            if true_val is not None:
                ax.axhline(true_val, color=COLOR_TRUE, linestyle="--", linewidth=1.5)
            ax.set_ylabel(ylabel)
        axes[-1].set_xlabel("Time (s)")
        title = "Convergence of the estimates"
        if true_values:
            title += " (dashed = true value)"
        if cutoffs["overall"] is not None:
            title += f"\nshaded = not yet within 3σ of the final estimate (settles at t={cutoffs['overall']:.3g} s)"
        axes[0].set_title(title)
        fig.tight_layout()
    return fig


def plot_posteriors(result, true_values=None):
    """Smoothed marginal posteriors for A0, T1/2, B, plus their correlation matrix."""
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        panels = [
            (result.A0_grid, result.A0_marginal, "A0 (cps)", result.A0, result.u_A0,
             true_values.get("A0") if true_values else None, 0),
            (result.half_life_grid, result.half_life_marginal, "T1/2 (s)", result.half_life, result.u_half_life,
             true_values.get("half_life") if true_values else None, 3),
            (result.background_grid, result.background_marginal, "B (cps)", result.background, result.u_background,
             true_values.get("background") if true_values else None, 2),
        ]
        for ax, (grid, marginal, xlabel, value, u, true_val, dec) in zip(axes, panels):
            ax.plot(grid, marginal, color=COLOR_BLUE, linewidth=2)
            ax.fill_between(grid, marginal, color=COLOR_BLUE, alpha=0.2)
            if true_val is not None:
                ax.axvline(true_val, color=COLOR_TRUE, linestyle="--", linewidth=1.5)
            ax.set_xlabel(xlabel)
            ax.set_title(f"{xlabel.split(' ')[0]} = {value:.{dec}f} ± {u:.{dec}f}")
        axes[0].set_ylabel("Posterior probability")

        diverging_cmap = LinearSegmentedColormap.from_list("diverging", ["#e34948", "#f0efec", "#2a78d6"])
        labels = ["A0", "T1/2", "B"]
        im = axes[3].imshow(result.corr, cmap=diverging_cmap, vmin=-1, vmax=1)
        axes[3].set_xticks(range(3))
        axes[3].set_xticklabels(labels)
        axes[3].set_yticks(range(3))
        axes[3].set_yticklabels(labels)
        axes[3].grid(False)
        for i in range(3):
            for j in range(3):
                txt_color = "white" if abs(result.corr[i, j]) > 0.6 else COLOR_TEXT
                axes[3].text(j, i, f"{result.corr[i, j]:.2f}", ha="center", va="center", color=txt_color, fontsize=10)
        axes[3].set_title("Correlation")
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        title = "Posterior marginals and covariance"
        if true_values:
            title += " (dashed = true value)"
        fig.suptitle(title)
        fig.tight_layout()
    return fig
