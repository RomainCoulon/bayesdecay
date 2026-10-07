"""Decay-curve model, data simulation, and the classical log-linear reference fit."""

from __future__ import annotations

import numpy as np

from .deadtime import apply_deadtime


def expected_counts(A0, lam, B, t_start, t_end):
    """Analytic integral of ``A0 * exp(-lam * t) + B`` between ``t_start`` and
    ``t_end`` (dead-time-free, i.e. the "true" expected count in that time span)."""
    return (A0 / lam) * (np.exp(-lam * t_start) - np.exp(-lam * t_end)) + B * (t_end - t_start)


def simulate_binned(A0, lam, B, t_max, tau_d, n_bins, dead_time_model="nonparalyzable", rng=None):
    """Simulate binned (histogrammed) counts directly, without enumerating individual
    events. Cost is O(n_bins), independent of the total number of counts -- this is
    what makes the estimator usable for measurements spanning hours or days, where
    enumerating every event would be computationally prohibitive.
    """
    rng = np.random.default_rng() if rng is None else rng
    bin_edges = np.linspace(0.0, t_max, n_bins + 1)
    t_start, t_end = bin_edges[:-1], bin_edges[1:]
    width = t_end - t_start

    mu_true = expected_counts(A0, lam, B, t_start, t_end)
    mu_obs = apply_deadtime(mu_true, width, tau_d, model=dead_time_model)
    counts = rng.poisson(mu_obs)
    return bin_edges, counts


def loglinear_fit(t_c, c, w, t_max_fallback):
    """Regress log(counts / channel width) against time, over non-empty channels.

    Gives a fast estimate of ``(A0, half_life)`` that ignores BOTH dead time and the
    background ``B`` (see the module-level comparison printed by the CLI). ``B`` is
    deliberately not estimated here: a late-acquisition plateau only reflects a real
    background once ``t_max`` extends well past the half-life -- when the acquisition
    is only on the order of one half-life, the "tail" is still decaying, and reading a
    plateau there would give an inflated, wrong background estimate. The full Bayesian
    estimator (see :mod:`bayesdecay.fit`) recovers B properly from the data and prior.

    Also returns the regression standard errors on A0 and half_life (propagated from
    the fitted-coefficient covariance, via ``numpy.polyfit(..., cov=True)`` and the
    delta method) -- used as the width of the weakly-informative Bayesian prior on
    A0 and half_life (see :mod:`bayesdecay.fit`).

    Returns
    -------
    A0_hat, half_life_hat, B_hat, sigma_A0, sigma_half_life
    """
    rate = c / w
    valid = c > 0
    if valid.sum() < 3:
        a0_fallback = 0.5 * max(rate.max(), 1.0)
        return a0_fallback, t_max_fallback, 0.0, 0.5 * a0_fallback, 0.5 * t_max_fallback
    (slope, intercept), pcov = np.polyfit(t_c[valid], np.log(rate[valid]), 1, cov=True)
    sigma_slope, sigma_intercept = np.sqrt(np.diag(pcov))
    lam_hat = max(-slope, 1e-12)
    half_life_hat = np.log(2) / lam_hat
    a0_hat = np.exp(intercept)
    # Delta method: half_life = ln2/lam, A0 = exp(intercept).
    sigma_half_life = (np.log(2) / lam_hat**2) * sigma_slope
    sigma_a0 = a0_hat * sigma_intercept
    return a0_hat, half_life_hat, 0.0, sigma_a0, sigma_half_life


def auto_bin_count(
    A0, lam, B, t_max, tau_d, dead_time_model="nonparalyzable",
    rate_change_tol=0.01, n_bins_prelim=200, n_bins_min=200, n_bins_max=20_000, rng=None,
):
    """Automatically choose the number of channels so that the dead-time formula's
    "rate approximately constant per channel" assumption holds to within
    ``rate_change_tol`` (relative).

    For a pure exponential ``A0 * exp(-lam * t)``, the RELATIVE rate of change is the
    constant ``-lam`` at every ``t`` (adding a constant background only damps this
    further at late times, so bounding with ``lam`` alone is a safe, conservative
    choice). Requiring ``lam * dt <= rate_change_tol`` over a channel of width ``dt``
    gives ``n_bins = ceil(t_max * ln2 / (rate_change_tol * half_life))``.

    Since the half-life is not known in advance, a cheap preliminary pass (coarse,
    just enough channels for a log-linear fit) gives a first estimate, which is then
    used to choose the channel count for the real analysis.

    Returns
    -------
    n_bins : int
    half_life_prelim : float
        The preliminary half-life estimate used to derive ``n_bins`` (useful for
        logging/diagnostics).
    """
    _, counts_prelim = simulate_binned(
        A0, lam, B, t_max, tau_d, n_bins_prelim, dead_time_model=dead_time_model, rng=rng
    )
    edges_prelim = np.linspace(0.0, t_max, n_bins_prelim + 1)
    tc_prelim = 0.5 * (edges_prelim[:-1] + edges_prelim[1:])
    w_prelim = np.diff(edges_prelim)
    return bin_count_from_coarse_pass(tc_prelim, counts_prelim, w_prelim, t_max, rate_change_tol, n_bins_min, n_bins_max)


def bin_count_from_coarse_pass(bin_centers, counts, bin_width, t_max, rate_change_tol=0.01, n_bins_min=200, n_bins_max=20_000):
    """Same channel-count rule as :func:`auto_bin_count`, but starting from an already
    binned coarse pass (real or simulated) rather than running a new simulation --
    used for real experimental data, where there is no "true" model to simulate from.

    Returns
    -------
    n_bins : int
    half_life_prelim : float
    """
    _, half_life_prelim, *_ = loglinear_fit(bin_centers, counts, bin_width, t_max)
    n_bins = int(np.clip(
        np.ceil(t_max * np.log(2) / (rate_change_tol * half_life_prelim)), n_bins_min, n_bins_max
    ))
    return n_bins, half_life_prelim
