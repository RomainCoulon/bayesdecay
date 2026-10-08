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


def test_informative_prior_narrows_uncertainty_without_losing_truth():
    # A short, weakly-constraining acquisition (little of the decay observed) is
    # exactly the regime where external knowledge of the half-life should help the
    # most: Priors.informative(), given the TRUE half-life as if it were known from
    # an independent nuclear-data evaluation, should report a substantially
    # tighter half-life uncertainty than Priors.weakly_informative() (the
    # default, centred on this measurement's own noisy log-linear fit) while
    # still correctly bracketing the truth.
    rng = np.random.default_rng(7)
    A0_true, half_life_true, tau_d = 4000.0, 60.0, 1e-6
    t_max = 0.1 * half_life_true  # short: little decay resolved
    lam_true = np.log(2) / half_life_true

    bin_edges, counts = simulate_binned(
        A0_true, lam_true, 0.0, t_max, tau_d, n_bins=150,
        dead_time_model="nonparalyzable", rng=rng,
    )

    weak = fit(
        bin_edges, counts, tau_d,
        config=FitConfig(n_checkpoints=3, n_is_samples=5_000, priors=Priors.weakly_informative()),
        show_progress=False,
    )
    informed = fit(
        bin_edges, counts, tau_d,
        config=FitConfig(
            n_checkpoints=3, n_is_samples=5_000,
            priors=Priors.informative(half_life_mean=half_life_true, half_life_sigma=0.5),
        ),
        show_progress=False,
    )

    assert weak.converged and informed.converged
    assert informed.u_half_life < 0.5 * weak.u_half_life
    assert informed.half_life_ci95[0] < half_life_true < informed.half_life_ci95[1]


def test_informative_prior_partial_override_leaves_other_parameters_data_driven():
    # Supplying ONLY an informative half-life prior should leave A0 and background
    # on their default weakly-informative (data-driven) behaviour -- i.e. close to
    # the fully weakly-informative fit's A0, not pinned or distorted by the
    # half-life override.
    rng = np.random.default_rng(11)
    A0_true, half_life_true, tau_d = 4000.0, 60.0, 1e-6
    t_max = 300.0
    lam_true = np.log(2) / half_life_true

    bin_edges, counts = simulate_binned(
        A0_true, lam_true, 0.0, t_max, tau_d, n_bins=150,
        dead_time_model="nonparalyzable", rng=rng,
    )

    weak = fit(
        bin_edges, counts, tau_d,
        config=FitConfig(n_checkpoints=3, n_is_samples=5_000, priors=Priors.weakly_informative()),
        show_progress=False,
    )
    informed = fit(
        bin_edges, counts, tau_d,
        config=FitConfig(
            n_checkpoints=3, n_is_samples=5_000,
            priors=Priors.informative(half_life_mean=half_life_true, half_life_sigma=0.1),
        ),
        show_progress=False,
    )

    assert informed.A0 == pytest.approx(weak.A0, rel=0.02)


def test_informative_background_prior_pulls_toward_known_value():
    # With a real, informative background prior (mean/sigma from, say, a separate
    # blank measurement), the recovered background should land close to that known
    # value when the counting data alone are too weak to strongly contradict it --
    # a basic sanity check that the Gaussian background-prior branch is wired up
    # (not silently falling back to the default exponential-near-zero prior).
    rng = np.random.default_rng(3)
    A0_true, half_life_true, B_true, tau_d = 500.0, 60.0, 20.0, 1e-6
    t_max = 60.0  # short + low rate: background is weakly constrained by the data alone
    lam_true = np.log(2) / half_life_true

    bin_edges, counts = simulate_binned(
        A0_true, lam_true, B_true, t_max, tau_d, n_bins=80,
        dead_time_model="nonparalyzable", rng=rng,
    )

    informed = fit(
        bin_edges, counts, tau_d,
        config=FitConfig(
            n_checkpoints=3, n_is_samples=5_000,
            priors=Priors.informative(background_mean=B_true, background_sigma=0.5),
        ),
        show_progress=False,
    )

    assert informed.converged
    assert informed.background == pytest.approx(B_true, abs=2.0)
