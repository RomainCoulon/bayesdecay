import numpy as np
import pytest

from bayesdecay.deadtime import apply_deadtime
from bayesdecay.model import expected_counts, loglinear_fit, simulate_binned


def test_expected_counts_matches_numerical_integration():
    A0, lam, B = 1000.0, np.log(2) / 50.0, 5.0
    t_start, t_end = np.array([3.0]), np.array([7.0])
    analytic = expected_counts(A0, lam, B, t_start, t_end)[0]

    t_fine = np.linspace(t_start[0], t_end[0], 200_001)
    rate = A0 * np.exp(-lam * t_fine) + B
    numeric = np.trapz(rate, t_fine)

    assert analytic == pytest.approx(numeric, rel=1e-6)


def test_expected_counts_is_additive_over_adjacent_intervals():
    A0, lam, B = 500.0, np.log(2) / 20.0, 1.0
    whole = expected_counts(A0, lam, B, np.array([0.0]), np.array([10.0]))[0]
    first_half = expected_counts(A0, lam, B, np.array([0.0]), np.array([4.0]))[0]
    second_half = expected_counts(A0, lam, B, np.array([4.0]), np.array([10.0]))[0]
    assert whole == pytest.approx(first_half + second_half, rel=1e-10)


def test_simulate_binned_shapes_and_nonnegativity():
    rng = np.random.default_rng(0)
    bin_edges, counts = simulate_binned(
        A0=2000.0, lam=np.log(2) / 30.0, B=2.0, t_max=100.0, tau_d=5e-6, n_bins=50, rng=rng
    )
    assert bin_edges.shape == (51,)
    assert counts.shape == (50,)
    assert np.all(counts >= 0)


def test_loglinear_fit_recovers_known_parameters_without_noise():
    # Use expected (noise-free) counts directly: the log-linear fit should recover
    # A0 and the half-life almost exactly when there's no background and no noise.
    A0, half_life = 3000.0, 40.0
    lam = np.log(2) / half_life
    edges = np.linspace(0.0, 400.0, 201)
    t_start, t_end = edges[:-1], edges[1:]
    width = t_end - t_start
    centers = 0.5 * (t_start + t_end)
    counts = expected_counts(A0, lam, 0.0, t_start, t_end)

    A0_hat, half_life_hat, B_hat, sigma_A0, sigma_t12 = loglinear_fit(centers, counts, width, 400.0)

    assert A0_hat == pytest.approx(A0, rel=1e-3)
    assert half_life_hat == pytest.approx(half_life, rel=1e-3)
    assert B_hat == 0.0
    assert sigma_A0 >= 0
    assert sigma_t12 >= 0


@pytest.mark.parametrize("dead_time_model", ["nonparalyzable", "paralyzable"])
def test_loglinear_fit_dead_time_correction_removes_bias(dead_time_model):
    # Even with zero background and zero noise, an UNCORRECTED log-linear fit on
    # dead-time-thinned counts reads a smaller A0 and a shorter half-life than truth
    # (dead-time losses are heaviest at the high rate right after t=0). Passing tau_d
    # should correct this back to (near) the true values.
    A0, half_life, tau_d = 50_000.0, 20.0, 20e-6  # high rate -> sizeable dead-time effect
    lam = np.log(2) / half_life
    edges = np.linspace(0.0, 200.0, 401)
    t_start, t_end = edges[:-1], edges[1:]
    width = t_end - t_start
    centers = 0.5 * (t_start + t_end)

    true_counts = expected_counts(A0, lam, 0.0, t_start, t_end)
    observed_counts = apply_deadtime(true_counts, width, tau_d, model=dead_time_model)

    A0_uncorrected, t12_uncorrected, *_ = loglinear_fit(centers, observed_counts, width, 200.0)
    A0_corrected, t12_corrected, *_ = loglinear_fit(
        centers, observed_counts, width, 200.0, tau_d=tau_d, dead_time_model=dead_time_model
    )

    assert A0_uncorrected < 0.95 * A0  # clearly biased low
    assert A0_corrected == pytest.approx(A0, rel=1e-3)
    assert t12_corrected == pytest.approx(half_life, rel=1e-3)
    assert abs(A0_corrected - A0) < abs(A0_uncorrected - A0)
    assert abs(t12_corrected - half_life) < abs(t12_uncorrected - half_life)
