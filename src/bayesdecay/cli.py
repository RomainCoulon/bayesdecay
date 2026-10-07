"""Command-line interface: ``bayesdecay simulate`` (validation on synthetic data) and
``bayesdecay fit`` (analysis of a real list-mode timestamp CSV file)."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from . import __version__
from .deadtime import DEAD_TIME_MODELS
from .fit import FitConfig, fit as run_fit
from .io import TIMESTAMP_UNITS, load_timestamps, timestamps_to_binned
from .model import auto_bin_count, simulate_binned
from .plotting import plot_activity, plot_convergence, plot_posteriors
from .priors import Priors
from .report import check_consistency, format_summary_table, save_results_csv


def _common_fit_args(parser):
    parser.add_argument("--dead-time", type=float, default=20e-6, help="Detector dead time, in seconds (default: 20e-6).")
    parser.add_argument(
        "--dead-time-model", choices=DEAD_TIME_MODELS, default="nonparalyzable",
        help="Dead-time model: 'nonparalyzable' (non-extending, the common case for most "
             "digital electronics) or 'paralyzable' (extending -- every true event, recorded "
             "or not, restarts the dead period). Default: nonparalyzable.",
    )
    parser.add_argument("--b-prior-scale", type=float, default=5.0, help="Mean of the exponential prior on background B, in cps (default: 5.0).")
    parser.add_argument("--prior-widen-k", type=float, default=5.0, help="Widening factor applied to the log-linear regression SE for the A0/T1/2 prior scale (default: 5.0).")
    parser.add_argument("--prior-df", type=float, default=4.0, help="Degrees of freedom of the Student-t prior on A0/T1/2 -- lower is less informative (heavier tails), higher approaches a Gaussian (default: 4.0).")
    parser.add_argument("--n-checkpoints", type=int, default=20, help="Number of points in the convergence trace (default: 20).")
    parser.add_argument("--n-is-samples", type=int, default=100_000, help="Importance-sampling draws for marginal smoothing (default: 100000).")
    parser.add_argument("--output-dir", default=".", help="Directory to write figures and the results CSV into (default: current directory).")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating and saving figures.")
    parser.add_argument("--show", action="store_true", help="Display figures interactively (in addition to saving them).")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress bars and stage messages.")


def _config_from_args(args):
    return FitConfig(
        dead_time_model=args.dead_time_model,
        n_checkpoints=args.n_checkpoints,
        n_is_samples=args.n_is_samples,
        priors=Priors(b_prior_scale=args.b_prior_scale, prior_widen_k=args.prior_widen_k, prior_df=args.prior_df),
    )


def _report_and_plot(result, bin_edges, counts, t_max, args, true_values=None):
    os.makedirs(args.output_dir, exist_ok=True)

    print()
    print(format_summary_table(result, true_values=true_values))
    print()

    csv_path = os.path.join(args.output_dir, "bayesdecay_results.csv")
    save_results_csv(result, csv_path, true_values=true_values)
    print(f"Results table saved: {csv_path}")

    if not result.converged:
        print(
            "WARNING: the local covariance grid could not be widened enough -- the "
            "reported uncertainties may be truncated. Consider relaxing the defaults "
            "in FitConfig if you are using the library API."
        )
    for warning in check_consistency(result):
        print(f"WARNING: {warning}")

    if args.no_plots:
        return

    figures = {
        "bayesdecay_activity.png": plot_activity(result, t_max, bin_edges, counts, true_values=true_values),
        "bayesdecay_convergence.png": plot_convergence(result, true_values=true_values),
        "bayesdecay_posteriors.png": plot_posteriors(result, true_values=true_values),
    }
    for name, fig in figures.items():
        path = os.path.join(args.output_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {path}")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


def _cmd_simulate(args):
    rng = np.random.default_rng(args.seed)
    lam = np.log(2) / args.half_life
    true_values = {"A0": args.a0, "half_life": args.half_life, "background": args.background}

    if not args.quiet:
        print("[1/3] Choosing channel count automatically...")
    n_bins, half_life_prelim = auto_bin_count(
        args.a0, lam, args.background, args.acquisition_time, args.dead_time,
        dead_time_model=args.dead_time_model, rng=rng,
    )
    if not args.quiet:
        print(f"  Preliminary T1/2 (coarse pass): {half_life_prelim:.4g} s")
        print(f"  -> n_bins = {n_bins} (channel width = {args.acquisition_time / n_bins:.4g} s)")
        print("[2/3] Simulating binned counting data...")
    bin_edges, counts = simulate_binned(
        args.a0, lam, args.background, args.acquisition_time, args.dead_time, n_bins,
        dead_time_model=args.dead_time_model, rng=rng,
    )
    print(f"Channels: {n_bins}, total simulated counts: {counts.sum()}")

    if not args.quiet:
        print("[3/3] Running Bayesian fit...")
    result = run_fit(bin_edges, counts, args.dead_time, config=_config_from_args(args), show_progress=not args.quiet)

    _report_and_plot(result, bin_edges, counts, args.acquisition_time, args, true_values=true_values)


def _cmd_fit(args):
    if not args.quiet:
        print("[1/3] Loading timestamps...")
    timestamps = load_timestamps(args.input, column=args.column, unit=args.timestamp_unit)
    print(f"Loaded {len(timestamps)} events spanning {timestamps[-1]:.6g} s.")

    if not args.quiet:
        print("[2/3] Choosing channel count automatically...")
    bin_edges, counts, half_life_prelim = timestamps_to_binned(
        timestamps, t_max=args.acquisition_time, n_bins=args.n_bins,
    )
    n_bins = len(counts)
    t_max = bin_edges[-1]
    if not args.quiet:
        if half_life_prelim is not None:
            print(f"  Preliminary T1/2 (coarse pass): {half_life_prelim:.4g} s")
        print(f"  -> n_bins = {n_bins} (channel width = {t_max / n_bins:.4g} s)")
    print(f"Channels: {n_bins}, total counts: {counts.sum()}")

    if not args.quiet:
        print("[3/3] Running Bayesian fit...")
    result = run_fit(bin_edges, counts, args.dead_time, config=_config_from_args(args), show_progress=not args.quiet)

    _report_and_plot(result, bin_edges, counts, t_max, args, true_values=None)


def build_parser():
    parser = argparse.ArgumentParser(prog="bayesdecay", description="Bayesian estimation of radioactive half-life, activity, and background from counting data.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_sim = subparsers.add_parser(
        "simulate",
        help="Simulate synthetic counting data and validate the estimator against known true values.",
    )
    p_sim.add_argument("--a0", type=float, default=5000.0, help="True initial activity, in cps (default: 5000).")
    p_sim.add_argument("--half-life", type=float, default=500.0, help="True half-life, in seconds (default: 500).")
    p_sim.add_argument("--background", type=float, default=0.0, help="True background, in cps (default: 0).")
    p_sim.add_argument("--acquisition-time", type=float, default=2000.0, help="Acquisition duration, in seconds (default: 2000).")
    p_sim.add_argument("--seed", type=int, default=None, help="Random seed, for reproducible simulations.")
    _common_fit_args(p_sim)
    p_sim.set_defaults(func=_cmd_simulate)

    p_fit = subparsers.add_parser(
        "fit",
        help="Fit real experimental list-mode timestamp data loaded from a CSV file.",
    )
    p_fit.add_argument("input", help="Path to a CSV file of event timestamps.")
    p_fit.add_argument("--column", default=None, help="Column name or 0-based index holding the timestamps (default: first column).")
    p_fit.add_argument(
        "--timestamp-unit", choices=sorted(TIMESTAMP_UNITS), default="seconds",
        help="Unit the timestamps are recorded in (default: seconds). Every other quantity "
             "(--dead-time, --acquisition-time) is in seconds, so get this right.",
    )
    p_fit.add_argument("--acquisition-time", type=float, default=None, help="Acquisition duration, in seconds (default: time of the last event).")
    p_fit.add_argument("--n-bins", type=int, default=None, help="Fixed channel count (default: chosen automatically).")
    _common_fit_args(p_fit)
    p_fit.set_defaults(func=_cmd_fit)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
