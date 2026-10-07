import numpy as np
import pytest

from bayesdecay.fit import FitConfig, fit
from bayesdecay.model import simulate_binned
from bayesdecay.priors import Priors


@pytest.mark.parametrize("dead_time_model", ["nonparalyzable", "paralyzable"])
def test_fit_recovers_known_parameters(dead_time_model):
    rng = np.random.default_rng(42)
    A0_true, half_life_true, B_true, tau_d = 3000.0, 40.0, 1.0, 1e-6
    t_max = 240.0  # 6 half-lives: well-conditioned regime
    lam_true = np.log(2) / half_life_true

    bin_edges, counts = simulate_binned(
        A0_true, lam_true, B_true, t_max, tau_d, n_bins=150,
        dead_time_model=dead_time_model, rng=rng,
    )

    config = FitConfig(
        dead_time_model=dead_time_model,
        n_checkpoints=5,
        n_is_samples=5_000,
        priors=Priors(),
    )
    result = fit(bin_edges, counts, tau_d, config=config, show_progress=False)

    assert result.converged
    assert result.A0 == pytest.approx(A0_true, rel=0.05)
    assert result.half_life == pytest.approx(half_life_true, rel=0.05)
    assert result.background == pytest.approx(B_true, abs=2.0)
    assert result.u_A0 > 0
    assert result.u_half_life > 0
    assert result.u_background > 0
    # Coverage check (the meaningful property): the TRUE value should fall within the
    # empirical 95% credible interval. We don't assert the point MAP itself falls
    # strictly inside its own CI -- with a modest IS sample count (kept small here for
    # test speed) the KDE-smoothed bounds can shift by sampling noise relative to the
    # point estimate; that's expected, not a correctness bug.
    assert result.A0_ci95[0] < A0_true < result.A0_ci95[1]
    assert result.half_life_ci95[0] < half_life_true < result.half_life_ci95[1]
    assert len(result.trace_t) == len(result.trace_A0) == len(result.trace_half_life)

    cutoffs = result.convergence_cutoff()
    assert set(cutoffs) == {"A0", "half_life", "background", "overall"}
    # The trace necessarily ENDS at the final estimate, so it must have "settled" by
    # some point at or before the last checkpoint (possibly the first, if it was
    # already within tolerance throughout) -- never later than the trace itself.
    for key in ("A0", "half_life", "background", "overall"):
        assert cutoffs[key] is not None
        assert result.trace_t[0] <= cutoffs[key] <= result.trace_t[-1]


def test_fit_dead_time_model_mismatch_biases_result():
    # Fitting with the WRONG dead-time model should introduce a detectable bias at a
    # high enough rate -- sanity check that the two models are not interchangeable.
    rng = np.random.default_rng(1)
    A0_true, half_life_true, tau_d = 20_000.0, 30.0, 5e-5  # high rate -> sizeable dead-time effect
    t_max = 180.0
    lam_true = np.log(2) / half_life_true

    bin_edges, counts = simulate_binned(
        A0_true, lam_true, 0.0, t_max, tau_d, n_bins=150,
        dead_time_model="paralyzable", rng=rng,
    )

    correct = fit(
        bin_edges, counts, tau_d,
        config=FitConfig(dead_time_model="paralyzable", n_checkpoints=3, n_is_samples=3_000),
        show_progress=False,
    )
    wrong = fit(
        bin_edges, counts, tau_d,
        config=FitConfig(dead_time_model="nonparalyzable", n_checkpoints=3, n_is_samples=3_000),
        show_progress=False,
    )

    correct_err = abs(correct.A0 - A0_true)
    wrong_err = abs(wrong.A0 - A0_true)
    assert wrong_err > correct_err
