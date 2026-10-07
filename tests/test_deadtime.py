import numpy as np
import pytest

from bayesdecay.deadtime import apply_deadtime, invert_deadtime


@pytest.mark.parametrize("model", ["nonparalyzable", "paralyzable"])
def test_zero_rate_is_fixed_point(model):
    assert apply_deadtime(np.array([0.0]), np.array([1.0]), tau_d=1e-5, model=model)[0] == 0.0


@pytest.mark.parametrize("model", ["nonparalyzable", "paralyzable"])
def test_low_rate_limit_matches_true_rate(model):
    # At rate*tau_d << 1, both models should barely perturb the true rate.
    width = 1.0
    true_counts = np.array([10.0])  # rate = 10 cps
    tau_d = 1e-6  # rate*tau_d = 1e-5, negligible
    observed = apply_deadtime(true_counts, width, tau_d, model=model)
    assert observed[0] == pytest.approx(10.0, rel=1e-4)


def test_nonparalyzable_is_monotonic_and_saturates():
    width = 1.0
    tau_d = 1e-3
    rates_true = np.array([10.0, 100.0, 1000.0, 1e6])
    observed = apply_deadtime(rates_true, width, tau_d, model="nonparalyzable") / width
    assert np.all(np.diff(observed) > 0)  # strictly increasing
    assert observed[-1] < 1.0 / tau_d  # saturates below 1/tau_d


def test_paralyzable_has_a_maximum_and_decreases_beyond_it():
    width = 1.0
    tau_d = 1e-3
    rates_true = np.linspace(1.0, 10.0 / tau_d, 2000)
    observed = apply_deadtime(rates_true, width, tau_d, model="paralyzable") / width
    peak_idx = np.argmax(observed)
    # Peak should sit near n = 1/tau_d, and the curve must fall off after it.
    assert rates_true[peak_idx] == pytest.approx(1.0 / tau_d, rel=0.05)
    assert observed[-1] < observed[peak_idx]


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        apply_deadtime(np.array([1.0]), np.array([1.0]), tau_d=1e-6, model="not-a-model")


@pytest.mark.parametrize("model", ["nonparalyzable", "paralyzable"])
def test_invert_deadtime_is_the_exact_inverse_below_the_paralysis_rate(model):
    tau_d = 1e-3
    rates_true = np.array([1.0, 50.0, 200.0, 500.0, 900.0])  # all well below 1/tau_d = 1000
    width = 1.0
    observed = apply_deadtime(rates_true, width, tau_d, model=model) / width
    recovered = invert_deadtime(observed, tau_d, model=model)
    assert recovered == pytest.approx(rates_true, rel=1e-6)


def test_invert_deadtime_unknown_model_raises():
    with pytest.raises(ValueError):
        invert_deadtime(np.array([1.0]), tau_d=1e-6, model="not-a-model")
