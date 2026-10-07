"""Manual, runnable smoke test of the bayesdecay PUBLIC PYTHON API.

This is deliberately separate from tests/ (the pytest suite CI runs): it exercises
the library the way a USER would import and call it -- `import bayesdecay`, nothing
internal -- as an end-to-end integration check and a living usage example, rather
than the unit-level checks in tests/. Not collected by pytest; run it directly:

    python dev/test_api.py

Exits non-zero (via assert) on the first thing that looks wrong; otherwise writes its
figures/CSVs to dev/output/ and prints a final summary. Requires the package to be
installed (``pip install -e .`` from the repo root) or this file run with the repo's
src/ on PYTHONPATH.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

import bayesdecay as bd
from bayesdecay import plotting

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def section(title):
    print()
    print(f"=== {title} ===")


def check_fit_matches_truth(result, true_values, label):
    for name, true_val in true_values.items():
        ci = getattr(result, f"{name}_ci95")
        # Same tolerance as bayesdecay.report.check_consistency: background (and any
        # parameter) has a physical floor at 0, and when the true value legitimately
        # sits there, the empirical CI's lower bound naturally lands a little above 0
        # (a one-sided pile-up at the boundary) -- not a sign anything is wrong.
        tol = 0.02 * (ci[1] - ci[0])
        assert ci[0] - tol <= true_val <= ci[1] + tol, (
            f"[{label}] {name}: true value {true_val} not inside 95% CI {ci} (tol={tol:.3g})"
        )
    print(f"[{label}] OK -- A0={result.A0:.1f}+/-{result.u_A0:.1f}, "
          f"T1/2={result.half_life:.2f}+/-{result.u_half_life:.2f}, "
          f"B={result.background:.2f}+/-{result.u_background:.2f}")


def simulate_and_fit(label, dead_time_model, A0=4000.0, half_life=60.0, background=1.0,
                      tau_d=10e-6, t_max=360.0, seed=0):
    section(label)
    rng = np.random.default_rng(seed)
    lam = np.log(2) / half_life

    # Auto-binning: the same rule used by `bayesdecay simulate`, derived from a cheap
    # preliminary pass rather than a hard-coded channel count.
    n_bins, half_life_prelim = bd.auto_bin_count(
        A0, lam, background, t_max, tau_d, dead_time_model=dead_time_model, rng=rng
    )
    print(f"auto_bin_count -> n_bins={n_bins} (preliminary T1/2 estimate {half_life_prelim:.3g} s)")

    bin_edges, counts = bd.simulate_binned(
        A0, lam, background, t_max, tau_d, n_bins, dead_time_model=dead_time_model, rng=rng
    )
    assert counts.sum() > 0, "simulation produced zero counts -- parameters are off"

    config = bd.FitConfig(
        dead_time_model=dead_time_model,
        n_checkpoints=10,
        n_is_samples=20_000,
        priors=bd.Priors(b_prior_scale=5.0, prior_widen_k=5.0, prior_df=4.0),
    )
    result = bd.fit(bin_edges, counts, tau_d, config=config, show_progress=False)
    assert result.converged, f"[{label}] local covariance grid did not converge"

    check_fit_matches_truth(
        result, {"A0": A0, "half_life": half_life, "background": background}, label
    )

    warnings = bd.check_consistency(result)
    for w in warnings:
        print(f"[{label}] consistency warning: {w}")

    cutoff = result.convergence_cutoff()["overall"]
    print(f"[{label}] convergence trace settles at t={cutoff}")

    true_values = {"A0": A0, "half_life": half_life, "background": background}
    print(bd.format_summary_table(result, true_values=true_values))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f"{label}_results.csv")
    bd.save_results_csv(result, csv_path, true_values=true_values)

    fig1 = plotting.plot_activity(result, t_max, bin_edges, counts, true_values=true_values)
    fig1.savefig(os.path.join(OUTPUT_DIR, f"{label}_activity.png"), dpi=110, bbox_inches="tight")
    fig2 = plotting.plot_convergence(result, true_values=true_values)
    fig2.savefig(os.path.join(OUTPUT_DIR, f"{label}_convergence.png"), dpi=110, bbox_inches="tight")
    fig3 = plotting.plot_posteriors(result, true_values=true_values)
    fig3.savefig(os.path.join(OUTPUT_DIR, f"{label}_posteriors.png"), dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    print(f"[{label}] wrote CSV + 3 figures to {OUTPUT_DIR}")

    return result


def test_deadtime_round_trip():
    section("deadtime: apply/invert round trip")
    for model in bd.DEAD_TIME_MODELS:
        tau_d = 2e-5
        n_true = np.array([100.0, 2_000.0, 20_000.0])
        m = bd.apply_deadtime(n_true, 1.0, tau_d, model=model)
        n_recovered = bd.invert_deadtime(m, tau_d, model=model)
        assert np.allclose(n_recovered, n_true, rtol=1e-4), f"{model}: invert_deadtime did not round-trip"
        print(f"{model}: round-trip OK")


def test_real_data_ingestion_path():
    section("real data: load_timestamps + timestamps_to_binned (CSV round trip)")
    A0, half_life, tau_d, t_max = 5000.0, 45.0, 15e-6, 270.0
    lam = np.log(2) / half_life
    rng = np.random.default_rng(2)

    # Build a pseudo list-mode CSV: histogram, then spread each channel's counts
    # uniformly within the channel -- good enough to exercise the ingestion path
    # end to end, not a physically exact list-mode simulator.
    n_bins, _ = bd.auto_bin_count(A0, lam, 0.0, t_max, tau_d, rng=rng)
    bin_edges, counts = bd.simulate_binned(A0, lam, 0.0, t_max, tau_d, n_bins, rng=rng)
    t_start, t_end = bin_edges[:-1], bin_edges[1:]
    events = np.sort(np.concatenate([
        rng.uniform(s, e, size=c) for c, s, e in zip(counts, t_start, t_end) if c > 0
    ]))

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "events.csv")
        with open(csv_path, "w") as f:
            f.write("timestamp_s\n")
            f.writelines(f"{t:.6f}\n" for t in events)

        timestamps = bd.load_timestamps(csv_path, unit="seconds")
        assert len(timestamps) == len(events)
        assert timestamps[0] == 0.0  # zeroed to the first event

        loaded_edges, loaded_counts, half_life_prelim = bd.timestamps_to_binned(
            timestamps, tau_d=tau_d, dead_time_model="nonparalyzable"
        )
    print(f"loaded {len(timestamps)} events, auto-binned to {len(loaded_counts)} channels "
          f"(preliminary T1/2={half_life_prelim:.3g} s)")

    result = bd.fit(loaded_edges, loaded_counts, tau_d, show_progress=False)
    check_fit_matches_truth(result, {"A0": A0, "half_life": half_life, "background": 0.0}, "real_data")


def main():
    test_deadtime_round_trip()
    test_real_data_ingestion_path()
    simulate_and_fit("nonparalyzable", dead_time_model="nonparalyzable")
    simulate_and_fit("paralyzable", dead_time_model="paralyzable", tau_d=25e-6)

    print()
    print("ALL API SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
